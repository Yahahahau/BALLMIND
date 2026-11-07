import { Quat, qMul, qX, qY, qZ, rotate, Vec3 } from '../../math/quat';
import type { HazardDef, MotionDef, PlatformDef, SpinDef } from './types';

/**
 * Pure, deterministic pose functions of simulation time. The physics world drives
 * its kinematic bodies from these, and the renderer / replay viewer evaluate the very
 * same functions – so a replay needs to store only the ball.
 */
export function motionOffset(m: MotionDef | undefined, t: number): Vec3 {
  if (!m) return [0, 0, 0];
  const s = m.amp * Math.sin((2 * Math.PI * t) / m.period + m.phase);
  return [m.axis[0] * s, m.axis[1] * s, m.axis[2] * s];
}

export const spinAngle = (s: SpinDef | undefined, t: number): number => (s ? s.phase + s.speed * t : 0);

export function platformRotation(p: PlatformDef, t: number): Quat {
  let q = qY(p.yaw + spinAngle(p.spin, t));
  if (p.tilt) q = qMul(qMul(q, qX(p.tilt[0])), qZ(p.tilt[1]));
  return q;
}

/** Body (cuboid) centre at time t. */
export function platformCenter(p: PlatformDef, t: number): Vec3 {
  const q = platformRotation(p, 0);
  const down = rotate(q, [0, -p.thick / 2, 0]);
  const o = motionOffset(p.motion, t);
  return [p.pos[0] + down[0] + o[0], p.pos[1] + down[1] + o[1], p.pos[2] + down[2] + o[2]];
}

/** Centre of the walkable top surface at time t. */
export function platformTop(p: PlatformDef, t: number): Vec3 {
  const q = platformRotation(p, t);
  const c = platformCenter(p, t);
  const up = rotate(q, [0, p.thick / 2, 0]);
  return [c[0] + up[0], c[1] + up[1], c[2] + up[2]];
}

export function hazardRotation(h: HazardDef, t: number): Quat {
  return qY(h.yaw + spinAngle(h.spin, t));
}
export function hazardCenter(h: HazardDef, t: number): Vec3 {
  const o = motionOffset(h.motion, t);
  return [h.pos[0] + o[0], h.pos[1] + o[1], h.pos[2] + o[2]];
}

const H = 1 / 60;
/** Linear velocity of the platform body centre (numerical derivative). */
export function platformVelocity(p: PlatformDef, t: number): Vec3 {
  if (!p.motion) return [0, 0, 0];
  const a = motionOffset(p.motion, t);
  const b = motionOffset(p.motion, t + H);
  return [(b[0] - a[0]) / H, (b[1] - a[1]) / H, (b[2] - a[2]) / H];
}
export function hazardVelocity(h: HazardDef, t: number): Vec3 {
  if (!h.motion) return [0, 0, 0];
  const a = motionOffset(h.motion, t);
  const b = motionOffset(h.motion, t + H);
  return [(b[0] - a[0]) / H, (b[1] - a[1]) / H, (b[2] - a[2]) / H];
}
