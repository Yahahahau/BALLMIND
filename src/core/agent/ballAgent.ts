import type { ActionDef } from '../action/actionSet';
import { FIXED_DT, PhysicsWorld, RAPIER } from '../physics/physicsWorld';

export interface BallParams {
  radius: number;
  mass: number;
  /** Horizontal acceleration (m/s²) when pushing on the ground. */
  accel: number;
  /** Fraction of `accel` available while airborne. */
  airControl: number;
  /** Upward impulse per unit mass. */
  jumpSpeed: number;
  /** Linear damping strength used by the "stop" action (1/s). */
  brake: number;
  /** Optional overrides (defaults keep the original Reach The Square feel). */
  restitution?: number;
  friction?: number;
  linearDamping?: number;
  angularDamping?: number;
}

export const DEFAULT_BALL: BallParams = {
  radius: 0.5,
  mass: 1,
  accel: 9,
  airControl: 0.35,
  jumpSpeed: 4.2,
  brake: 7,
};

/**
 * The physical AI agent: a dynamic rigid body with a sphere collider.
 * The brain controls it ONLY through forces/impulses – never by writing positions
 * (except `teleport`, used between episodes).
 */
export class BallAgent {
  readonly body: RAPIER.RigidBody;
  readonly collider: RAPIER.Collider;
  grounded = false;

  constructor(
    private readonly physics: PhysicsWorld,
    readonly params: BallParams,
    start: [number, number, number],
  ) {
    const { body, collider } = physics.createDynamicSphere(start, params.radius, {
      mass: params.mass,
      friction: params.friction ?? 0.9,
      restitution: params.restitution ?? 0.3,
      linearDamping: params.linearDamping ?? 0.6,
      angularDamping: params.angularDamping ?? 0.25,
    });
    this.body = body;
    this.collider = collider;
  }

  /** Apply one fixed step worth of control. `jumpNow` triggers the jump impulse (once per decision). */
  applyAction(action: ActionDef, jumpNow: boolean): void {
    const p = this.params;
    this.grounded = this.physics.groundedBelow(this.body, p.radius + 0.07);
    const control = this.grounded ? 1 : p.airControl;
    const k = p.accel * p.mass * FIXED_DT * control;
    let ix = action.move[0] * k;
    let iz = action.move[1] * k;
    let iy = 0;

    if (action.jump && jumpNow && this.grounded) iy = p.jumpSpeed * p.mass;

    if (action.brake) {
      const v = this.body.linvel();
      const w = this.body.angvel();
      const kb = Math.min(1, p.brake * FIXED_DT) * p.mass;
      ix -= v.x * kb;
      iz -= v.z * kb;
      const inertia = 0.4 * p.mass * p.radius * p.radius;
      const kt = Math.min(1, p.brake * FIXED_DT) * inertia;
      this.body.applyTorqueImpulse({ x: -w.x * kt, y: -w.y * kt, z: -w.z * kt }, true);
    }

    if (ix !== 0 || iy !== 0 || iz !== 0) this.body.applyImpulse({ x: ix, y: iy, z: iz }, true);
  }

  /** Used only to start a new episode. */
  teleport(x: number, y: number, z: number): void {
    this.body.setTranslation({ x, y, z }, true);
    this.body.setRotation({ x: 0, y: 0, z: 0, w: 1 }, true);
    this.body.setLinvel({ x: 0, y: 0, z: 0 }, true);
    this.body.setAngvel({ x: 0, y: 0, z: 0 }, true);
  }
}
