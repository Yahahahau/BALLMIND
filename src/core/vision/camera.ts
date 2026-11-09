import type { VisionSettings } from './config';

export interface VisionCameraPose {
  eye: [number, number, number];
  right: [number, number, number];
  up: [number, number, number];
  fwd: [number, number, number];
  /** tan(fov/2) */
  tanHalf: number;
}

/**
 * The AI camera: independent of the player camera, always centred on the ball.
 * World-aligned (it follows the ball's position but does not roll with it).
 */
export function visionCameraPose(s: VisionSettings, ballX: number, ballY: number, ballZ: number): VisionCameraPose {
  const th = (s.pitch * Math.PI) / 180;
  const yaw = (s.yaw * Math.PI) / 180;
  const hx = Math.sin(yaw);
  const hz = -Math.cos(yaw);
  const ct = Math.cos(th);
  const st = Math.sin(th);
  const fwd: [number, number, number] = [hx * ct, -st, hz * ct];
  const up: [number, number, number] = [hx * st, ct, hz * st];
  // right = fwd × up
  const right: [number, number, number] = [
    fwd[1] * up[2] - fwd[2] * up[1],
    fwd[2] * up[0] - fwd[0] * up[2],
    fwd[0] * up[1] - fwd[1] * up[0],
  ];
  const rl = Math.hypot(right[0], right[1], right[2]) || 1;
  right[0] /= rl;
  right[1] /= rl;
  right[2] /= rl;
  const eye: [number, number, number] = [ballX - fwd[0] * s.distance, ballY - fwd[1] * s.distance, ballZ - fwd[2] * s.distance];
  return { eye, right, up, fwd, tanHalf: Math.tan((s.fov * Math.PI) / 360) };
}
