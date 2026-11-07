import { PhysicsWorld, RAPIER } from '../../physics/physicsWorld';
import { OBBY } from './config';

export interface Underfoot {
  /** Collider handle of what is under the ball, or -1. */
  handle: number;
  /** Vertical distance from ball centre to the surface, or Infinity. */
  dist: number;
  nx: number;
  nz: number;
}

/**
 * Ray sensors: terrain probes (is there floor ahead, how high), obstacle rays at ball
 * height and an under-foot ray. All results are raw metres – normalisation is done by
 * the observation builder.
 */
export class Sensors {
  readonly probes = new Float32Array(OBBY.probeDirs * OBBY.probeDist.length);
  readonly obstacles = new Float32Array(OBBY.probeDirs);
  readonly under: Underfoot = { handle: -1, dist: Infinity, nx: 0, nz: 0 };
  private readonly ray = new RAPIER.Ray({ x: 0, y: 0, z: 0 }, { x: 0, y: -1, z: 0 });
  private readonly dirs: [number, number][] = [];

  constructor() {
    for (let k = 0; k < OBBY.probeDirs; k++) {
      const a = (k / OBBY.probeDirs) * Math.PI * 2;
      this.dirs.push([Math.cos(a), Math.sin(a)]);
    }
  }

  private cast(physics: PhysicsWorld, body: RAPIER.RigidBody, ox: number, oy: number, oz: number, dx: number, dy: number, dz: number, max: number) {
    const r = this.ray;
    r.origin.x = ox;
    r.origin.y = oy;
    r.origin.z = oz;
    r.dir.x = dx;
    r.dir.y = dy;
    r.dir.z = dz;
    return physics.world.castRay(r, max, true, RAPIER.QueryFilterFlags.EXCLUDE_SENSORS, undefined, undefined, body);
  }

  sense(physics: PhysicsWorld, body: RAPIER.RigidBody): void {
    const t = body.translation();
    const n = OBBY.probeDist.length;
    // terrain probes: ray down from 1.5 m above the ball, value = surface height relative to the ball's feet
    for (let k = 0; k < this.dirs.length; k++) {
      for (let j = 0; j < n; j++) {
        const d = OBBY.probeDist[j];
        const hit = this.cast(physics, body, t.x + this.dirs[k][0] * d, t.y + 1.5, t.z + this.dirs[k][1] * d, 0, -1, 0, 12);
        // -1 = nothing there (void); otherwise (surface - feet) / 3 clamped
        this.probes[k * n + j] = hit ? Math.max(-1, Math.min(1, (t.y + 1.5 - hit.timeOfImpact - (t.y - 0.5)) / 3)) : -1;
      }
      const o = this.cast(physics, body, t.x, t.y, t.z, this.dirs[k][0], 0, this.dirs[k][1], OBBY.rayRange);
      this.obstacles[k] = o ? 1 - o.timeOfImpact / OBBY.rayRange : 0;
    }
    // under-foot ray with surface normal
    this.ray.origin.x = t.x;
    this.ray.origin.y = t.y;
    this.ray.origin.z = t.z;
    this.ray.dir.x = 0;
    this.ray.dir.y = -1;
    this.ray.dir.z = 0;
    const u = physics.world.castRayAndGetNormal(this.ray, 9, true, RAPIER.QueryFilterFlags.EXCLUDE_SENSORS, undefined, undefined, body);
    if (u) {
      this.under.handle = u.collider.handle;
      this.under.dist = u.timeOfImpact;
      this.under.nx = u.normal.x;
      this.under.nz = u.normal.z;
    } else {
      this.under.handle = -1;
      this.under.dist = Infinity;
      this.under.nx = this.under.nz = 0;
    }
  }
}
