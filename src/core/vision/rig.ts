import type { EnvSnapshot } from '../task/types';
import type { VisionSettings } from './config';

/**
 * A "vision rig" turns the current world into the AI camera's RGB image.
 * Output layout: planar CHW, uint8, row 0 = top: [R plane | G plane | B plane], size 3·H·W.
 */
export interface VisionRig {
  readonly kind: string;
  settings: VisionSettings;
  render(snap: EnvSnapshot, out: Uint8Array): void;
  /** Change camera/size settings (may reallocate GPU targets). */
  configure(s: VisionSettings): void;
  dispose(): void;
}

/** Shared flat colours of the AI view (unlit: the AI never sees shadows, AO or sky lighting). */
export const VISION_COLORS = {
  void: [18, 22, 36] as [number, number, number],
  floor: [96, 112, 140] as [number, number, number],
  target: [60, 230, 110] as [number, number, number],
  ball: [255, 140, 40] as [number, number, number],
};
