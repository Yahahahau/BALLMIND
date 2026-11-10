/**
 * The AI's own looking direction. The camera yaw/pitch live here (not in the physics body) and are changed
 * ONLY by the AI's gaze actions – nothing points the camera at the target.
 */
export interface GazeAction {
  id: string;
  label: string;
  /** yaw direction (+1 = look right/clockwise from above), pitch direction (+1 = up) */
  dyaw: number;
  dpitch: number;
}

const D = 1;
export const GAZE_ACTIONS: GazeAction[] = [
  { id: 'hold', label: 'Maintain Gaze', dyaw: 0, dpitch: 0 },
  { id: 'left', label: 'Look Left', dyaw: -D, dpitch: 0 },
  { id: 'right', label: 'Look Right', dyaw: D, dpitch: 0 },
  { id: 'up', label: 'Look Up', dyaw: 0, dpitch: D },
  { id: 'down', label: 'Look Down', dyaw: 0, dpitch: -D },
  { id: 'up_left', label: 'Look Up-Left', dyaw: -D, dpitch: D },
  { id: 'up_right', label: 'Look Up-Right', dyaw: D, dpitch: D },
  { id: 'down_left', label: 'Look Down-Left', dyaw: -D, dpitch: -D },
  { id: 'down_right', label: 'Look Down-Right', dyaw: D, dpitch: -D },
];

export const PITCH_LIMIT = 80; // degrees

export class GazeController {
  /** degrees; 0 = looking along world -Z, positive turns right (towards +X). Wraps at ±180. */
  yaw = 0;
  /** degrees, clamped to ±80 */
  pitch = 0;
  /** degrees of change per second while a gaze action is held */
  yawRate: number;
  pitchRate: number;

  constructor(yawRate = 120, pitchRate = 80) {
    this.yawRate = yawRate;
    this.pitchRate = pitchRate;
  }

  reset(yaw = 0, pitch = 0): void {
    this.yaw = wrapDeg(yaw);
    this.pitch = clampPitch(pitch);
  }

  /** Advance the gaze by `dt` seconds under gaze action `a` (smooth, incremental). Roll does not exist. */
  step(a: number, dt: number): void {
    const g = GAZE_ACTIONS[a];
    if (!g) return;
    this.yaw = wrapDeg(this.yaw + g.dyaw * this.yawRate * dt);
    this.pitch = clampPitch(this.pitch + g.dpitch * this.pitchRate * dt);
  }
}

export const wrapDeg = (d: number): number => {
  let x = (d + 180) % 360;
  if (x < 0) x += 360;
  return x - 180;
};
export const clampPitch = (p: number): number => Math.max(-PITCH_LIMIT, Math.min(PITCH_LIMIT, p));
