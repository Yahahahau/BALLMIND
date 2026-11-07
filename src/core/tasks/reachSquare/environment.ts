import { createLocomotionActions } from '../../action/actionSet';
import { BallAgent, DEFAULT_BALL } from '../../agent/ballAgent';
import { Rng } from '../../math/rng';
import { initPhysics, PhysicsWorld, RAPIER } from '../../physics/physicsWorld';
import type { BodyPose, Environment, EnvSnapshot, StepResult } from '../../task/types';
import { REACH_CONFIG as C } from './config';
import { buildObservation, OBSERVATION_SIZE, ReachState } from './observation';
import { computeReward } from './reward';

/** REACH THE SQUARE: touch the target tile on the floor without falling off. */
export class ReachSquareEnvironment implements Environment {
  readonly observationSize = OBSERVATION_SIZE;
  readonly actionSet = createLocomotionActions();
  readonly substepsPerDecision = C.substepsPerDecision;
  readonly maxDecisions = C.maxDecisions;

  private readonly physics: PhysicsWorld;
  private readonly ball: BallAgent;
  private readonly targetBody: RAPIER.RigidBody;
  private readonly targetCollider: RAPIER.Collider;
  private readonly rng: Rng;
  private readonly obs = new Float32Array(OBSERVATION_SIZE);
  private readonly state: ReachState = { px: 0, py: 0, pz: 0, vx: 0, vy: 0, vz: 0, tx: 0, tz: 0, grounded: false };
  private target: [number, number] = [0, 0];
  private prevDist = 0;
  private epoch = 0;
  private lastActionJumped = false;

  constructor(seed: number) {
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
    buildObservation(this.obs, this.state);
    return this.obs;
  }

  physicsStep(action: number, firstSubstep: boolean): void {
    const def = this.actionSet.get(action);
    this.ball.applyAction(def, firstSubstep);
    if (firstSubstep) this.lastActionJumped = def.jump;
    this.physics.step();
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
    buildObservation(this.obs, s);
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
