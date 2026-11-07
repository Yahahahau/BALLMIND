import { BallAgent } from '../../agent/ballAgent';
import { Rng } from '../../math/rng';
import { FIXED_DT, initPhysics, PhysicsWorld, RAPIER } from '../../physics/physicsWorld';
import type { BodyPose, Environment, EnvSnapshot, EpisodeSummary, ReplayTrace, StepResult, TaskNotice } from '../../task/types';
import { createObbyActions } from './actions';
import { OBBY, REWARD } from './config';
import { Curriculum } from './curriculum';
import { hazardCenter, hazardRotation, hazardVelocity, platformCenter, platformRotation, platformTop, platformVelocity, spinAngle } from './kinematics';
import { STAGES } from './levels/compile';
import { buildObbyObservation, HAZARD_STRIDE, OBS_SIZE, ObbyObsInput } from './observation';
import { Sensors } from './sensors';
import type { CompiledStage } from './types';

interface KBody {
  body: RAPIER.RigidBody;
  kinematic: boolean;
}

interface Run {
  startIdx: number;
  lifeK: number;
  banked: boolean[];
  cpPassed: boolean[];
  respawnIdx: number;
  falls: number;
  decisions: number;
  bestP: number;
  prevP: number;
  sinceImprove: number;
  partial: boolean;
  finished: boolean;
}

export interface ObbyControl {
  startMode: 'stage' | 'checkpoint';
  /** 1-based checkpoint number used when startMode = 'checkpoint' (manual stage mode only). */
  startCheckpoint: number;
}

/**
 * OBBY CHALLENGE environment. The ball is a real Rapier rigid body driven only by
 * impulses; platforms/hazards are kinematic bodies moved by pure functions of time,
 * so friction carries the ball on moving and rotating platforms.
 */
export class ObbyEnvironment implements Environment {
  readonly observationSize = OBS_SIZE;
  readonly actionSet = createObbyActions();
  readonly substepsPerDecision = OBBY.substepsPerDecision;
  readonly curriculum = new Curriculum(STAGES.length, STAGES.map((s) => s.def.parTime));
  control: ObbyControl = { startMode: 'stage', startCheckpoint: 1 };

  private physics!: PhysicsWorld;
  private ball!: BallAgent;
  private readonly sensors = new Sensors();
  private readonly rng: Rng;
  private stageIdx = -1;
  private stage!: CompiledStage;
  private plats: KBody[] = [];
  private hazards: KBody[] = [];
  private handleToPlat = new Map<number, number>();
  private simTime = 0;
  private t0 = 0;
  private epoch = 0;
  private run!: Run;
  private airTime = 0;
  private grounded = false;
  private readonly obs = new Float32Array(OBS_SIZE);
  private readonly hazardBuf = new Float32Array(OBBY.hazardSlots * HAZARD_STRIDE);
  private notices: TaskNotice[] = [];
  private tracePos: number[] = [];
  private traceEvents: number[][] = [];
  private lastTrace: ReplayTrace | null = null;
  private lastSummary: EpisodeSummary | null = null;
  private dyn = new Float32Array(0);

  constructor(seed: number) {
    this.rng = new Rng(seed * 7717 + 3);
  }

  get maxDecisions(): number {
    return this.stage ? this.stage.def.maxDecisions : STAGES[0].def.maxDecisions;
  }

  // ------------------------------------------------------------------ world construction

  private buildStage(idx: number): void {
    this.physics?.dispose();
    this.stageIdx = idx;
    this.stage = STAGES[idx];
    this.physics = new PhysicsWorld(-9.81);
    this.plats = [];
    this.hazards = [];
    this.handleToPlat.clear();
    const w = this.physics.world;
    this.stage.def.platforms.forEach((p, i) => {
      const kinematic = !!(p.motion || p.spin);
      const c = platformCenter(p, 0);
      const q = platformRotation(p, 0);
      const desc = (kinematic ? RAPIER.RigidBodyDesc.kinematicPositionBased() : RAPIER.RigidBodyDesc.fixed())
        .setTranslation(c[0], c[1], c[2])
        .setRotation({ x: q[0], y: q[1], z: q[2], w: q[3] });
      const body = w.createRigidBody(desc);
      const cd = RAPIER.ColliderDesc.cuboid(p.size[0] / 2, p.thick / 2, p.size[1] / 2).setFriction(p.friction).setRestitution(0.05);
      if (p.friction < 0.3) cd.setFrictionCombineRule(RAPIER.CoefficientCombineRule.Min);
      const col = w.createCollider(cd, body);
      this.handleToPlat.set(col.handle, i);
      this.plats.push({ body, kinematic });
    });
    for (const h of this.stage.def.hazards) {
      const c = hazardCenter(h, 0);
      const q = hazardRotation(h, 0);
      const body = w.createRigidBody(
        RAPIER.RigidBodyDesc.kinematicPositionBased().setTranslation(c[0], c[1], c[2]).setRotation({ x: q[0], y: q[1], z: q[2], w: q[3] }),
      );
      const col = w.createCollider(RAPIER.ColliderDesc.cuboid(h.size[0] / 2, h.size[1] / 2, h.size[2] / 2).setFriction(0.5).setRestitution(0.1), body);
      this.handleToPlat.set(col.handle, -2);
      this.hazards.push({ body, kinematic: true });
    }
    this.ball = new BallAgent(this.physics, OBBY.ball, [0, 1, 0]);
    this.dyn = new Float32Array((this.plats.filter((p) => p.kinematic).length + this.hazards.length) * 7);
  }

  private poseKinematics(t: number, teleport: boolean): void {
    const defs = this.stage.def;
    defs.platforms.forEach((p, i) => {
      const k = this.plats[i];
      if (!k.kinematic) return;
      const c = platformCenter(p, t);
      const q = platformRotation(p, t);
      if (teleport) {
        k.body.setTranslation({ x: c[0], y: c[1], z: c[2] }, true);
        k.body.setRotation({ x: q[0], y: q[1], z: q[2], w: q[3] }, true);
      }
      k.body.setNextKinematicTranslation({ x: c[0], y: c[1], z: c[2] });
      k.body.setNextKinematicRotation({ x: q[0], y: q[1], z: q[2], w: q[3] });
    });
    defs.hazards.forEach((h, i) => {
      const k = this.hazards[i];
      const c = hazardCenter(h, t);
      const q = hazardRotation(h, t);
      if (teleport) {
        k.body.setTranslation({ x: c[0], y: c[1], z: c[2] }, true);
        k.body.setRotation({ x: q[0], y: q[1], z: q[2], w: q[3] }, true);
      }
      k.body.setNextKinematicTranslation({ x: c[0], y: c[1], z: c[2] });
      k.body.setNextKinematicRotation({ x: q[0], y: q[1], z: q[2], w: q[3] });
    });
  }

  // ------------------------------------------------------------------ episode lifecycle

  reset(): Float32Array {
    this.epoch++;
    const idx = this.curriculum.chooseStage(this.rng);
    if (idx !== this.stageIdx) this.buildStage(idx);

    let startIdx = 0;
    if (!this.curriculum.config.auto && this.control.startMode === 'checkpoint') {
      const cp = this.stage.checkpointIdx[this.control.startCheckpoint - 1];
      if (cp !== undefined) startIdx = cp;
    }
    this.t0 = this.rng.range(0, 60);
    this.simTime = this.t0;
    this.poseKinematics(this.simTime, true);

    const n = this.stage.def.platforms.length;
    const banked = Array.from({ length: n }, (_, i) => i <= startIdx);
    const cpPassed = Array.from({ length: n }, (_, i) => i <= startIdx && this.stage.def.platforms[i].kind === 'checkpoint');
    this.run = {
      startIdx,
      lifeK: startIdx,
      banked,
      cpPassed,
      respawnIdx: startIdx,
      falls: 0,
      decisions: 0,
      bestP: 0,
      prevP: 0,
      sinceImprove: 0,
      partial: startIdx > 0,
      finished: false,
    };
    this.placeBallAt(startIdx);
    this.airTime = 0;
    this.grounded = true;
    this.run.prevP = this.stage.cum[startIdx];
    this.run.bestP = this.run.prevP;
    this.tracePos = [];
    this.traceEvents = [];
    this.pushTrace();
    this.lastSummary = null;
    this.measure();
    return this.obs;
  }

  private placeBallAt(platformIdx: number): void {
    const p = this.stage.def.platforms[platformIdx];
    const top = platformTop(p, this.simTime);
    this.ball.teleport(top[0], top[1] + OBBY.ball.radius + OBBY.respawnLift, top[2]);
  }

  private pushTrace(): void {
    const t = this.ball.body.translation();
    this.tracePos.push(Math.round(t.x * 100), Math.round(t.y * 100), Math.round(t.z * 100));
  }

  physicsStep(action: number, firstSubstep: boolean): void {
    this.poseKinematics(this.simTime + FIXED_DT, false);
    this.ball.applyAction(this.actionSet.get(action), firstSubstep);
    this.physics.step();
    this.simTime += FIXED_DT;
  }

  // ------------------------------------------------------------------ measurement + reward

  /** Path progress (metres) from the ball position and the furthest platform landed in this life. */
  private progressOf(bx: number, by: number, bz: number): number {
    const cum = this.stage.cum;
    const k = this.run.lifeK;
    const plats = this.stage.def.platforms;
    if (k >= plats.length - 1) return cum[plats.length - 1];
    const a = plats[k].pos;
    const b = plats[k + 1].pos;
    const dx = b[0] - a[0];
    const dy = (b[1] - a[1]) * 0.5;
    const dz = b[2] - a[2];
    const L2 = dx * dx + dy * dy + dz * dz;
    const t = L2 > 1e-9 ? ((bx - a[0]) * dx + (by - OBBY.ball.radius - a[1]) * 0.5 * dy + (bz - a[2]) * dz) / L2 : 0;
    return cum[k] + Math.max(0, Math.min(1, t)) * (cum[k + 1] - cum[k]);
  }

  private measure(): void {
    const b = this.ball.body;
    const t = b.translation();
    const v = b.linvel();
    const w = b.angvel();
    this.sensors.sense(this.physics, b);
    const u = this.sensors.under;
    const supIdx = u.handle >= 0 ? (this.handleToPlat.get(u.handle) ?? -1) : -1;
    this.grounded = u.dist <= OBBY.ball.radius + 0.12;
    const def = this.stage.def;
    const T = this.simTime;
    const n = def.platforms.length;
    const feetY = t.y - OBBY.ball.radius;

    // support platform velocity (moving: linear, rotating: omega x r)
    let sv: [number, number, number] = [0, 0, 0];
    let slippery = false;
    if (this.grounded && supIdx >= 0) {
      const p = def.platforms[supIdx];
      slippery = p.friction < 0.3;
      if (p.motion) sv = platformVelocity(p, T);
      if (p.spin) {
        const c = platformTop(p, T);
        sv = [p.spin.speed * (t.z - c[2]), 0, -p.spin.speed * (t.x - c[0])];
      }
    }

    const nextIdx = Math.min(n - 1, this.run.lifeK + 1);
    const next2Idx = Math.min(n - 1, this.run.lifeK + 2);
    const np = def.platforms[nextIdx];
    const nt = platformTop(np, T);
    const n2t = platformTop(def.platforms[next2Idx], T);
    const nv = platformVelocity(np, T);

    // nearest hazards
    this.hazardBuf.fill(0);
    const order = def.hazards
      .map((h, i) => ({ i, d: Math.hypot(hazardCenter(h, T)[0] - t.x, hazardCenter(h, T)[2] - t.z) }))
      .filter((o) => o.d < 14)
      .sort((a, c) => a.d - c.d)
      .slice(0, OBBY.hazardSlots);
    order.forEach((o, slot) => {
      const h = def.hazards[o.i];
      const c = hazardCenter(h, T);
      const base = slot * HAZARD_STRIDE;
      let vx = 0;
      let vz = 0;
      let s2 = 0;
      let c2 = 0;
      let om = 0;
      if (h.spin) {
        om = h.spin.speed;
        vx = om * (t.z - c[2]);
        vz = -om * (t.x - c[0]);
        const th = h.yaw + spinAngle(h.spin, T);
        s2 = Math.sin(2 * th);
        c2 = Math.cos(2 * th);
      } else {
        const hv = hazardVelocity(h, T);
        vx = hv[0];
        vz = hv[2];
      }
      const cl = (x: number) => Math.max(-1.5, Math.min(1.5, x));
      this.hazardBuf[base] = 1;
      this.hazardBuf[base + 1] = cl((c[0] - t.x) / 10);
      this.hazardBuf[base + 2] = cl((c[2] - t.z) / 10);
      this.hazardBuf[base + 3] = cl(vx / 6);
      this.hazardBuf[base + 4] = cl(vz / 6);
      this.hazardBuf[base + 5] = s2;
      this.hazardBuf[base + 6] = c2;
      this.hazardBuf[base + 7] = cl(om / 3);
    });

    const cpTotal = this.stage.checkpointIdx.length;
    const input: ObbyObsInput = {
      vel: [v.x, v.y, v.z],
      ang: [w.x, w.y, w.z],
      grounded: this.grounded,
      airTime: this.airTime,
      underDist: u.dist,
      supportVel: sv,
      supportSlippery: slippery,
      normalX: u.nx,
      normalZ: u.nz,
      next: [nt[0] - t.x, nt[1] - feetY, nt[2] - t.z],
      next2: [n2t[0] - t.x, n2t[1] - feetY, n2t[2] - t.z],
      nextVel: nv,
      nextSize: np.size,
      nextMoving: !!np.motion,
      nextRotating: !!np.spin,
      progress: Math.min(1, this.progressOf(t.x, t.y, t.z) / this.stage.total),
      checkpointFrac: cpTotal ? this.run.cpPassed.filter(Boolean).length / cpTotal : 1,
      fallFrac: this.run.falls / OBBY.maxFalls,
      probes: this.sensors.probes,
      obstacles: this.sensors.obstacles,
      hazards: this.hazardBuf,
    };
    buildObbyObservation(this.obs, input);
  }

  evaluate(decisionIndex: number): StepResult {
    const run = this.run;
    const st = this.stage;
    const t = this.ball.body.translation();
    run.decisions = decisionIndex + 1;
    this.measure(); // fresh sensors (also provides observation for the non-fall case)
    const u = this.sensors.under;
    const supIdx = u.handle >= 0 ? (this.handleToPlat.get(u.handle) ?? -1) : -1;
    this.airTime = this.grounded ? 0 : this.airTime + 0.1;

    let reward = -REWARD.timePenalty;
    let success = false;
    let fell = false;
    let respawned = false;

    // --- landing / checkpoint / finish (physical contact only) ---
    if (this.grounded && supIdx >= 0 && supIdx > run.lifeK) {
      for (let k = run.lifeK + 1; k <= supIdx; k++) {
        if (!run.banked[k]) {
          run.banked[k] = true;
          reward += REWARD.landing;
          this.traceEvents.push([decisionIndex, 1, k]);
        }
      }
      run.lifeK = supIdx;
    }
    if (this.grounded && supIdx >= 0) {
      const kind = st.def.platforms[supIdx].kind;
      if (kind === 'checkpoint' && !run.cpPassed[supIdx]) {
        run.cpPassed[supIdx] = true;
        run.respawnIdx = supIdx;
        reward += REWARD.checkpoint;
        this.traceEvents.push([decisionIndex, 2, supIdx]);
      }
      if (supIdx === st.finishIdx) {
        success = true;
        run.finished = true;
        reward += REWARD.finish;
        this.traceEvents.push([decisionIndex, 4, 0]);
      }
    }

    // --- progress shaping: potential difference, so any loop (also fall + respawn) sums to zero ---
    let P = this.progressOf(t.x, t.y, t.z);
    reward += (P - run.prevP) * REWARD.progress;
    run.prevP = P;
    if (P > run.bestP + 1e-6) {
      run.bestP = P;
      run.sinceImprove = 0;
    } else run.sinceImprove++;

    // --- falling ---
    if (!success && t.y < st.voidY) {
      fell = true;
      run.falls++;
      reward += REWARD.fall;
      this.traceEvents.push([decisionIndex, 3, Math.round(P * 10)]);
      const respawnP = st.cum[run.respawnIdx];
      reward += (respawnP - P) * REWARD.progress; // potential drops back to the respawn point
      if (run.falls < OBBY.maxFalls) {
        run.lifeK = run.respawnIdx;
        this.placeBallAt(run.respawnIdx);
        this.airTime = 0;
        run.prevP = respawnP;
        P = respawnP;
        respawned = true;
        this.measure(); // observation of the respawned state
      }
    }
    this.pushTrace();

    const outOfFalls = fell && run.falls >= OBBY.maxFalls;
    const timeout = decisionIndex + 1 >= this.maxDecisions || run.sinceImprove >= OBBY.stallLimit;
    const done = success || outOfFalls || timeout;
    if (done) this.finishEpisode();
    return { reward, done, terminal: success || fell, success, failed: fell, respawned, obs: this.obs };
  }

  private finishEpisode(): void {
    const run = this.run;
    const st = this.stage;
    const progress = Math.min(1, run.bestP / st.total);
    this.lastSummary = {
      stage: this.stageIdx,
      progress: run.finished ? 1 : progress,
      checkpoints: run.cpPassed.filter(Boolean).length,
      falls: run.falls,
      partial: run.partial,
    };
    if (!run.partial) {
      const unlocked = this.curriculum.record(this.stageIdx, {
        success: run.finished,
        progress: this.lastSummary.progress,
        time: (run.decisions * OBBY.substepsPerDecision) * FIXED_DT,
      });
      if (unlocked !== null) this.notices.push({ kind: 'unlock', stage: unlocked, text: `Stage ${unlocked + 1} unlocked: ${STAGES[unlocked].def.name}` });
    }
    const pos = Int16Array.from(this.tracePos);
    this.lastTrace = { stage: this.stageIdx, t0: this.t0, startIdx: run.startIdx, pos, events: this.traceEvents };
  }

  // ------------------------------------------------------------------ optional extensions

  episodeSummary(): EpisodeSummary {
    return this.lastSummary ?? { stage: this.stageIdx, progress: 0, checkpoints: 0, falls: 0, partial: false };
  }

  takeTrace(): ReplayTrace | null {
    const t = this.lastTrace;
    this.lastTrace = null;
    return t;
  }

  drainNotices(): TaskNotice[] {
    const n = this.notices;
    this.notices = [];
    return n;
  }

  liveStats(): Record<string, unknown> {
    const st = this.stage ?? STAGES[0];
    const run = this.run;
    return {
      stage: Math.max(0, this.stageIdx),
      stageName: st.def.name,
      difficulty: st.def.difficulty,
      checkpoints: run ? run.cpPassed.filter(Boolean).length : 0,
      checkpointTotal: st.checkpointIdx.length,
      progress: run ? Math.min(1, run.prevP / st.total) : 0,
      bestProgress: run ? Math.min(1, run.bestP / st.total) : 0,
      runFalls: run ? run.falls : 0,
      partial: run ? run.partial : false,
      unlocked: this.curriculum.unlocked,
      curriculum: { ...this.curriculum.config },
      unlockProgress: STAGES.map((_, i) => this.curriculum.unlockProgress(i)),
      startMode: this.control.startMode,
      startCheckpoint: this.control.startCheckpoint,
      checkpointCounts: STAGES.map((s) => s.checkpointIdx.length),
    };
  }

  command(cmd: string, arg: unknown): void {
    const a = (arg ?? {}) as Record<string, unknown>;
    if (cmd === 'curriculum') this.curriculum.config = { ...this.curriculum.config, ...(a as object) };
    else if (cmd === 'resetProgress') this.curriculum.resetProgress();
    else if (cmd === 'start') this.control = { ...this.control, ...(a as Partial<ObbyControl>) };
  }

  exportState(): unknown {
    return { curriculum: this.curriculum.exportState(), control: this.control };
  }

  importState(s: unknown): void {
    const st = s as { curriculum?: ReturnType<Curriculum['exportState']>; control?: ObbyControl };
    if (st?.curriculum) this.curriculum.importState(st.curriculum);
    if (st?.control) this.control = { ...this.control, ...st.control };
  }

  snapshot(): EnvSnapshot {
    const p = this.ball.body.translation();
    const r = this.ball.body.rotation();
    const ball: BodyPose = { p: [p.x, p.y, p.z], q: [r.x, r.y, r.z, r.w] };
    let o = 0;
    const put = (b: RAPIER.RigidBody) => {
      const tr = b.translation();
      const ro = b.rotation();
      this.dyn[o++] = tr.x;
      this.dyn[o++] = tr.y;
      this.dyn[o++] = tr.z;
      this.dyn[o++] = ro.x;
      this.dyn[o++] = ro.y;
      this.dyn[o++] = ro.z;
      this.dyn[o++] = ro.w;
    };
    for (const k of this.plats) if (k.kinematic) put(k.body);
    for (const k of this.hazards) put(k.body);
    const v = this.ball.body.linvel();
    return {
      epoch: this.epoch,
      bodies: { ball },
      dyn: this.dyn.slice(),
      info: {
        stage: this.stageIdx,
        simTime: this.simTime,
        speed: Math.hypot(v.x, v.z),
        grounded: this.grounded ? 1 : 0,
      },
    };
  }

  dispose(): void {
    this.physics?.dispose();
  }
}

export async function createObbyEnvironment(seed: number): Promise<Environment> {
  await initPhysics();
  return new ObbyEnvironment(seed);
}

