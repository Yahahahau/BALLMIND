import { createLocomotionActions, type ActionDef } from '../../action/actionSet';
import { BallAgent, DEFAULT_BALL } from '../../agent/ballAgent';
import { Rng } from '../../math/rng';
import { initPhysics, PhysicsWorld, RAPIER } from '../../physics/physicsWorld';
import type { BodyPose, Environment, EnvSnapshot, StepResult } from '../../task/types';
import { REACH_CONFIG as C } from './config';
import { buildFpProprioception, buildObservation, buildProprioception, FP_PROPRIO_SIZE, OBSERVATION_SIZE, PROPRIO_SIZE, ReachState } from './observation';
import { computeReward } from './reward';

/** REACH THE SQUARE: touch the target tile on the floor without falling off. */
export class ReachSquareEnvironment implements Environment {
  readonly observationSize: number;
  readonly actionSet = createLocomotionActions();
  readonly substepsPerDecision = C.substepsPerDecision;
  readonly maxDecisions = C.maxDecisions;

  private readonly physics: PhysicsWorld;
  private readonly ball: BallAgent;
  private readonly targetBody: RAPIER.RigidBody;
  private readonly targetCollider: RAPIER.Collider;
  private readonly rng: Rng;
  private readonly obs: Float32Array;
  private readonly state: ReachState = { px: 0, py: 0, pz: 0, vx: 0, vy: 0, vz: 0, tx: 0, tz: 0, grounded: false };
  private target: [number, number] = [0, 0];
  private prevDist = 0;
  private epoch = 0;
  private lastActionJumped = false;

  /** `vision`: observation = physics sensors only (no coordinates); the target is found through the camera. */
  constructor(seed: number, private readonly vision: false | 'top' | 'fp' = false) {
    this.observationSize = vision === 'fp' ? FP_PROPRIO_SIZE : vision ? PROPRIO_SIZE : OBSERVATION_SIZE;
    this.obs = new Float32Array(this.observationSize);
    this.rng = new Rng(seed * 104729 + 7);
    this.physics = new PhysicsWorld(-9.81);
    const t = C.floorThickness / 2;
    this.physics.createStaticBox([0, -t, 0], [C.arenaHalf, t, C.arenaHalf], 0.9);
    // Sensor volume matches the visible 6 cm tile exactly, so "touching" means the sphere really meets the tile.
    const sensor = this.physics.createSensorBox([0, C.tileHeight / 2, 0], [C.targetHalf, C.tileHeight / 2, C.targetHalf]);
    this.targetBody = sensor.body;
    this.targetCollider = sensor.collider;
    this.ball = new BallAgent(this.physics, DEFAULT_BALL, C.start);
  }

  reset(): Float32Array {
    this.epoch++;
    this.ball.teleport(...C.start);
    // Random target inside the field, never on top of the start position.
    let tx = 0;
    let tz = 0;
    for (let i = 0; i < 50; i++) {
      tx = this.rng.range(-C.spawnHalf, C.spawnHalf);
      tz = this.rng.range(-C.spawnHalf, C.spawnHalf);
      if (Math.hypot(tx - C.start[0], tz - C.start[2]) >= C.minSpawnDistance) break;
    }
    this.target = [tx, tz];
    this.targetBody.setTranslation({ x: tx, y: C.tileHeight / 2, z: tz }, true);
    this.lastActionJumped = false;
    this.readState();
    this.prevDist = Math.hypot(this.state.tx - this.state.px, this.state.tz - this.state.pz);
    this.observe();
    return this.obs;
  }

  private yawDeg = 0;
  private pitchDeg = 0;
  private rot: ActionDef = { id: 'rot', label: 'rot', move: [0, 0], jump: false, brake: false };

  /** First-person: movement actions are relative to where the AI is looking. */
  setHeading(yawDeg: number, pitchDeg = this.pitchDeg): void {
    this.yawDeg = yawDeg;
    this.pitchDeg = pitchDeg;
  }

  physicsStep(action: number, firstSubstep: boolean): void {
    let def = this.actionSet.get(action);
    if (this.vision === 'fp') {
      const y = (this.yawDeg * Math.PI) / 180;
      const fwd = -def.move[1];
      const r = this.rot;
      r.move[0] = def.move[0] * Math.cos(y) + fwd * Math.sin(y);
      r.move[1] = def.move[0] * Math.sin(y) - fwd * Math.cos(y);
      r.jump = def.jump;
      r.brake = def.brake;
      def = r;
    }
    this.ball.applyAction(def, firstSubstep);
    if (firstSubstep) this.lastActionJumped = def.jump;
    this.physics.step();
  }

  private observe(): void {
    if (this.vision === 'fp') buildFpProprioception(this.obs, this.state, this.ball.body.angvel(), this.yawDeg, this.pitchDeg);
    else if (this.vision) buildProprioception(this.obs, this.state, this.ball.body.angvel());
    else buildObservation(this.obs, this.state);
  }

  private readState(): void {
    const s = this.state;
    const p = this.ball.body.translation();
    const v = this.ball.body.linvel();
    s.px = p.x;
    s.py = p.y;
    s.pz = p.z;
    s.vx = v.x;
    s.vy = v.y;
    s.vz = v.z;
    s.tx = this.target[0];
    s.tz = this.target[1];
    s.grounded = this.physics.groundedBelow(this.ball.body, C.ballRadius + 0.07);
  }

  evaluate(decisionIndex: number): StepResult {
    this.readState();
    const s = this.state;
    const dist = Math.hypot(s.tx - s.px, s.tz - s.pz);
    // Success = the ball's collider really touches the target volume.
    const success = this.physics.intersects(this.ball.collider, this.targetCollider);
    const fell = s.py < C.fallY;
    const timeout = decisionIndex + 1 >= this.maxDecisions;
    const reward = computeReward({
      prevDist: this.prevDist,
      dist,
      success,
      fell,
      jumped: this.lastActionJumped,
    });
    this.prevDist = dist;
    this.observe();
    const terminal = success || fell;
    return { reward, done: terminal || timeout, terminal, success, failed: fell, obs: this.obs };
  }

  snapshot(): EnvSnapshot {
    const p = this.ball.body.translation();
    const r = this.ball.body.rotation();
    const v = this.ball.body.linvel();
    const ball: BodyPose = { p: [p.x, p.y, p.z], q: [r.x, r.y, r.z, r.w] };
    const target: BodyPose = { p: [this.target[0], 0, this.target[1]], q: [0, 0, 0, 1] };
    return {
      epoch: this.epoch,
      bodies: { ball, target },
      info: { speed: Math.hypot(v.x, v.z), grounded: this.ball.grounded ? 1 : 0, dist: this.prevDist },
    };
  }

  dispose(): void {
    this.physics.dispose();
  }
}

export async function createReachSquareEnvironment(seed: number): Promise<Environment> {
  await initPhysics();
  return new ReachSquareEnvironment(seed);
}

export async function createReachVisionEnvironment(seed: number): Promise<Environment> {
  await initPhysics();
  return new ReachSquareEnvironment(seed, 'top');
}

export async function createReachFpEnvironment(seed: number): Promise<Environment> {
  await initPhysics();
  return new ReachSquareEnvironment(seed, 'fp');
}
