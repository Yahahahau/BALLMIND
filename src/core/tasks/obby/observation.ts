import { OBBY } from './config';

/** Everything the observation needs, already measured by the environment. */
export interface ObbyObsInput {
  vel: [number, number, number];
  ang: [number, number, number];
  grounded: boolean;
  airTime: number;
  /** Distance from ball centre to the surface below (Infinity = void). */
  underDist: number;
  supportVel: [number, number, number];
  supportSlippery: boolean;
  normalX: number;
  normalZ: number;
  /** Relative to the ball: next waypoint platform surface centre, and the one after. */
  next: [number, number, number];
  next2: [number, number, number];
  nextVel: [number, number, number];
  nextSize: [number, number];
  nextMoving: boolean;
  nextRotating: boolean;
  progress: number;
  checkpointFrac: number;
  fallFrac: number;
  probes: Float32Array;
  obstacles: Float32Array;
  /** Flattened hazard slots (HAZARD_STRIDE values each). */
  hazards: Float32Array;
}

export const HAZARD_STRIDE = 8;
const PROBES = OBBY.probeDirs * OBBY.probeDist.length;
export const OBS_SIZE = 3 + 3 + 4 + 3 + 1 + 2 + 3 + 1 + 2 + 3 + 3 + 2 + 2 + 3 + PROBES + OBBY.probeDirs + OBBY.hazardSlots * HAZARD_STRIDE;

const cl = (v: number, l = 1.5) => (v < -l ? -l : v > l ? l : v);

/**
 * Observation System. Fixed layout, every value normalised to about [-1, 1].
 *  vel(3) angVel(3) grounded airTime heightAbove void | supportVel(3) slippery normal(2)
 *  next platform rel(3) dist dir(2) | next2 rel(3) | nextVel(3) size(2) moving rotating
 *  progress checkpoint falls | terrain probes(8 dirs × 3 ranges) | obstacle rays(8) | hazards(2 × 8)
 */
export function buildObbyObservation(out: Float32Array, s: ObbyObsInput): void {
  let i = 0;
  out[i++] = cl(s.vel[0] / 12);
  out[i++] = cl(s.vel[1] / 10);
  out[i++] = cl(s.vel[2] / 12);
  out[i++] = cl(s.ang[0] / 25);
  out[i++] = cl(s.ang[1] / 25);
  out[i++] = cl(s.ang[2] / 25);
  out[i++] = s.grounded ? 1 : 0;
  out[i++] = Math.min(1, s.airTime);
  const hasGround = isFinite(s.underDist);
  out[i++] = hasGround ? Math.min(1, Math.max(0, (s.underDist - 0.5) / 4)) : 1;
  out[i++] = hasGround ? 0 : 1;
  out[i++] = cl(s.supportVel[0] / 6);
  out[i++] = cl(s.supportVel[1] / 6);
  out[i++] = cl(s.supportVel[2] / 6);
  out[i++] = s.supportSlippery ? 1 : 0;
  out[i++] = cl(s.normalX * 3);
  out[i++] = cl(s.normalZ * 3);
  out[i++] = cl(s.next[0] / 12);
  out[i++] = cl(s.next[1] / 4);
  out[i++] = cl(s.next[2] / 12);
  const hd = Math.hypot(s.next[0], s.next[2]);
  out[i++] = cl(hd / 12);
  out[i++] = hd > 1e-5 ? s.next[0] / hd : 0;
  out[i++] = hd > 1e-5 ? s.next[2] / hd : 0;
  out[i++] = cl(s.next2[0] / 20);
  out[i++] = cl(s.next2[1] / 6);
  out[i++] = cl(s.next2[2] / 20);
  out[i++] = cl(s.nextVel[0] / 6);
  out[i++] = cl(s.nextVel[1] / 6);
  out[i++] = cl(s.nextVel[2] / 6);
  out[i++] = cl(s.nextSize[0] / 8);
  out[i++] = cl(s.nextSize[1] / 8);
  out[i++] = s.nextMoving ? 1 : 0;
  out[i++] = s.nextRotating ? 1 : 0;
  out[i++] = s.progress;
  out[i++] = s.checkpointFrac;
  out[i++] = s.fallFrac;
  for (let k = 0; k < PROBES; k++) out[i++] = s.probes[k];
  for (let k = 0; k < OBBY.probeDirs; k++) out[i++] = s.obstacles[k];
  for (let k = 0; k < s.hazards.length; k++) out[i++] = s.hazards[k];
  if (i !== OBS_SIZE) throw new Error(`observation layout mismatch ${i} != ${OBS_SIZE}`);
}
