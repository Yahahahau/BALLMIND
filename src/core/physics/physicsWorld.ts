import RAPIER from '@dimforge/rapier3d-compat';

export { RAPIER };

let initPromise: Promise<void> | null = null;

/** Initialise the Rapier WASM module once (works in browsers, workers and Node). */
export function initPhysics(): Promise<void> {
  if (!initPromise) initPromise = RAPIER.init();
  return initPromise;
}

export const FIXED_DT = 1 / 60;

/** Thin wrapper around a Rapier world with a fixed time step. */
export class PhysicsWorld {
  readonly world: RAPIER.World;

  constructor(gravityY = -9.81) {
    this.world = new RAPIER.World({ x: 0, y: gravityY, z: 0 });
    this.world.timestep = FIXED_DT;
  }

  step(): void {
    this.world.step();
  }

  createStaticBox(center: [number, number, number], half: [number, number, number], friction = 0.8) {
    const body = this.world.createRigidBody(RAPIER.RigidBodyDesc.fixed().setTranslation(...center));
    const collider = this.world.createCollider(
      RAPIER.ColliderDesc.cuboid(...half)
        .setFriction(friction)
        .setRestitution(0.2),
      body,
    );
    return { body, collider };
  }

  /** A fixed trigger volume; objects pass through it but intersections are reported. */
  createSensorBox(center: [number, number, number], half: [number, number, number]) {
    const body = this.world.createRigidBody(RAPIER.RigidBodyDesc.fixed().setTranslation(...center));
    const collider = this.world.createCollider(RAPIER.ColliderDesc.cuboid(...half).setSensor(true), body);
    return { body, collider };
  }

  createDynamicSphere(
    center: [number, number, number],
    radius: number,
    opts: { mass: number; friction: number; restitution: number; linearDamping: number; angularDamping: number },
  ) {
    const body = this.world.createRigidBody(
      RAPIER.RigidBodyDesc.dynamic()
        .setTranslation(...center)
        .setLinearDamping(opts.linearDamping)
        .setAngularDamping(opts.angularDamping)
        .setCcdEnabled(true),
    );
    const collider = this.world.createCollider(
      RAPIER.ColliderDesc.ball(radius)
        .setMass(opts.mass)
        .setFriction(opts.friction)
        .setRestitution(opts.restitution),
      body,
    );
    return { body, collider };
  }

  /** True if the collider is resting on (or very near) a solid surface directly beneath it. */
  groundedBelow(body: RAPIER.RigidBody, reach: number): boolean {
    const t = body.translation();
    const ray = new RAPIER.Ray({ x: t.x, y: t.y, z: t.z }, { x: 0, y: -1, z: 0 });
    const hit = this.world.castRay(
      ray,
      reach,
      true,
      RAPIER.QueryFilterFlags.EXCLUDE_SENSORS,
      undefined,
      undefined,
      body,
    );
    return hit !== null;
  }

  intersects(a: RAPIER.Collider, b: RAPIER.Collider): boolean {
    return this.world.intersectionPair(a, b);
  }

  dispose(): void {
    this.world.free();
  }
}
