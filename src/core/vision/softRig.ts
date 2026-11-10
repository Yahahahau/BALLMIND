import type { EnvSnapshot } from '../task/types';
import { REACH_CONFIG as C } from '../tasks/reachSquare/config';
import { visionCameraPose } from './camera';
import type { VisionSettings } from './config';
import { FP_ABYSS_Y, FP_VIEW_DIST, VISION_COLORS, type VisionRig } from './rig';
import type { VisionCameraPose } from './camera';

/**
 * CPU reference camera for the Reach scene: casts one ray per pixel through the SAME camera model
 * as the GPU renderer (floor square, target tile, ball sphere, flat colours). It exists so the whole
 * perception→learning pipeline can be tested headlessly in Node (no WebGL there); it is not used in
 * the browser unless WebGL2/OffscreenCanvas are unavailable (graceful fallback, slower).
 */
export class SoftReachRig implements VisionRig {
  readonly kind = 'cpu';
  settings: VisionSettings;

  constructor(settings: VisionSettings) {
    this.settings = settings;
  }

  configure(s: VisionSettings): void {
    this.settings = s;
  }

  render(snap: EnvSnapshot, out: Uint8Array, pose?: VisionCameraPose): void {
    const s = this.settings;
    const n = s.size;
    const plane = n * n;
    const b = snap.bodies.ball.p;
    const t = snap.bodies.target.p;
    const fp = !!pose; // first-person: the AI's own body is not part of its scene
    const cam = pose ?? visionCameraPose(s, b[0], b[1], b[2]);
    const [ex, ey, ez] = cam.eye;
    const R = C.ballRadius;
    const top = C.tileHeight;
    const col = VISION_COLORS;
    for (let j = 0; j < n; j++) {
      const v = ((j + 0.5) / n) * 2 - 1;
      for (let i = 0; i < n; i++) {
        const u = ((i + 0.5) / n) * 2 - 1;
        let dx = cam.fwd[0] + cam.right[0] * u * cam.tanHalf - cam.up[0] * v * cam.tanHalf;
        let dy = cam.fwd[1] + cam.right[1] * u * cam.tanHalf - cam.up[1] * v * cam.tanHalf;
        let dz = cam.fwd[2] + cam.right[2] * u * cam.tanHalf - cam.up[2] * v * cam.tanHalf;
        const dl = Math.hypot(dx, dy, dz);
        dx /= dl;
        dy /= dl;
        dz /= dl;
        let c: [number, number, number] = fp ? col.sky : col.void;
        let best = Infinity;
        // floor (y = 0)
        if (dy < -1e-6) {
          const tt = -ey / dy;
          if (tt > 0) {
            const px = ex + dx * tt;
            const pz = ez + dz * tt;
            if (Math.abs(px) <= C.arenaHalf && Math.abs(pz) <= C.arenaHalf) {
              best = tt;
              c = col.floor;
            }
          }
          // target tile top (y = tileHeight)
          const t2 = (top - ey) / dy;
          if (t2 > 0 && t2 < best) {
            const px = ex + dx * t2;
            const pz = ez + dz * t2;
            if (Math.abs(px - t[0]) <= C.targetHalf && Math.abs(pz - t[2]) <= C.targetHalf) {
              best = t2;
              c = col.target;
            }
          }
        }
        if (fp && c === col.sky && dy < -1e-6 && (FP_ABYSS_Y - ey) / dy <= FP_VIEW_DIST) c = col.void;
        // ball sphere
        if (!fp) {
        const ox = ex - b[0];
        const oy = ey - b[1];
        const oz = ez - b[2];
        const bq = ox * dx + oy * dy + oz * dz;
        const cq = ox * ox + oy * oy + oz * oz - R * R;
        const disc = bq * bq - cq;
        if (disc >= 0) {
          const t3 = -bq - Math.sqrt(disc);
          if (t3 > 0 && t3 < best) {
            best = t3;
            c = col.ball;
          }
        }
        }
        const p = j * n + i;
        out[p] = c[0];
        out[plane + p] = c[1];
        out[2 * plane + p] = c[2];
      }
    }
  }

  dispose(): void {}
}
