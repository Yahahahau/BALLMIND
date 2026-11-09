import { REACH_CONFIG as C } from './config';

/** Raw (un-normalised) world state the observation is derived from. */
export interface ReachState {
  px: number;
  py: number;
  pz: number;
  vx: number;
  vy: number;
  vz: number;
  tx: number;
  tz: number;
  grounded: boolean;
}

export const OBSERVATION_SIZE = 17;

const clamp = (v: number, lim = 3) => (v < -lim ? -lim : v > lim ? lim : v);

/**
 * Observation System: every value is normalised to roughly [-1, 1].
 *  0-1  target offset (x,z)          2  distance            3-4 unit direction to target
 *  5-6  velocity (x,z)               7  speed toward target 8   sideways speed
 *  9-10 ball position (x,z)          11 distance to edge    12  vertical velocity
 *  13   touching ground              14 height              15  outside the arena
 *  16   speed
 */
export function buildObservation(out: Float32Array, s: ReachState): void {
  const span = C.arenaHalf * 2;
  const dx = s.tx - s.px;
  const dz = s.tz - s.pz;
  const dist = Math.hypot(dx, dz);
  const ux = dist > 1e-6 ? dx / dist : 0;
  const uz = dist > 1e-6 ? dz / dist : 0;
  const along = s.vx * ux + s.vz * uz;
  const side = s.vx * uz - s.vz * ux;
  const edge = Math.min(C.arenaHalf - Math.abs(s.px), C.arenaHalf - Math.abs(s.pz));
  const outside = Math.abs(s.px) > C.arenaHalf || Math.abs(s.pz) > C.arenaHalf;

  out[0] = clamp(dx / span);
  out[1] = clamp(dz / span);
  out[2] = clamp(dist / (span * Math.SQRT2));
  out[3] = ux;
  out[4] = uz;
  out[5] = clamp(s.vx / C.vMax);
  out[6] = clamp(s.vz / C.vMax);
  out[7] = clamp(along / C.vMax);
  out[8] = clamp(side / C.vMax);
  out[9] = clamp(s.px / C.arenaHalf);
  out[10] = clamp(s.pz / C.arenaHalf);
  out[11] = clamp(edge / C.arenaHalf);
  out[12] = clamp(s.vy / 8);
  out[13] = s.grounded ? 1 : 0;
  out[14] = clamp((s.py - C.ballRadius) / 2);
  out[15] = outside ? 1 : 0;
  out[16] = clamp(Math.hypot(s.vx, s.vz) / C.vMax);
}

/**
 * Vision-mode observation: ONLY the ball's own physics sensors (proprioception). No target
 * position, no ball position, no distances – the AI has to get those from its camera image.
 *  0-2 linear velocity (x,y,z)   3-5 angular velocity   6 touching ground   7 speed
 */
export const PROPRIO_SIZE = 8;

export function buildProprioception(
  out: Float32Array,
  s: ReachState,
  angVel: { x: number; y: number; z: number },
): void {
  out[0] = clamp(s.vx / C.vMax);
  out[1] = clamp(s.vy / 8);
  out[2] = clamp(s.vz / C.vMax);
  out[3] = clamp(angVel.x / 20);
  out[4] = clamp(angVel.y / 20);
  out[5] = clamp(angVel.z / 20);
  out[6] = s.grounded ? 1 : 0;
  out[7] = clamp(Math.hypot(s.vx, s.vz) / C.vMax);
}
