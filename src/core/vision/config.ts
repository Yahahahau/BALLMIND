import type { ConvSpec, VisionNetSpec } from './visionNet';

/** User/agent-facing vision settings. Everything here is a real parameter of the camera or the network input. */
export interface VisionSettings {
  /** Square AI camera image size in pixels (RGB). */
  size: number;
  /** Frames stacked into one network input (temporal perception). 1 = single image. */
  stack: number;
  /** Vertical field of view of the AI camera, degrees. */
  fov: number;
  /** Camera pitch below the horizon, degrees (90 = straight down). */
  pitch: number;
  /** Camera heading, degrees (world aligned – the camera does not roll with the ball). */
  yaw: number;
  /** Distance from the ball to the camera along its view direction, metres. */
  distance: number;
  /** 'top' = v1 bird's-eye camera; 'fp' = first-person eye camera (v2) with AI-controlled gaze. */
  mode: 'top' | 'fp';
  /** fp: gaze turn speed, degrees per second while a look action is held. */
  yawRate: number;
  pitchRate: number;
}

export const DEFAULT_VISION: VisionSettings = { size: 40, stack: 2, fov: 60, pitch: 90, yaw: 0, distance: 10, mode: 'top', yawRate: 120, pitchRate: 80 };

export const VISION_LIMITS = {
  size: [16, 128],
  stack: [1, 4],
  fov: [30, 110],
  pitch: [20, 90],
  yaw: [-180, 180],
  distance: [4, 30],
  yawRate: [30, 360],
  pitchRate: [20, 240],
} as const;

export function sanitizeVision(v: Partial<VisionSettings> | undefined, base: VisionSettings = DEFAULT_VISION): VisionSettings {
  const out = { ...base, ...(v ?? {}) } as VisionSettings;
  for (const k of Object.keys(VISION_LIMITS) as (keyof typeof VISION_LIMITS)[]) {
    const [lo, hi] = VISION_LIMITS[k];
    const n = Number(out[k]);
    out[k] = Math.min(hi, Math.max(lo, Number.isFinite(n) ? n : base[k]));
  }
  out.size = Math.round(out.size);
  out.stack = Math.round(out.stack);
  return out;
}

/** The CNN visual encoder layout for a given image size. */
export function encoderConvs(size: number): { convs: ConvSpec[]; feat: number } {
  if (size >= 96) {
    return {
      convs: [
        { k: 6, s: 3, out: 8 },
        { k: 4, s: 2, out: 16 },
        { k: 3, s: 2, out: 16 },
      ],
      feat: 96,
    };
  }
  return {
    convs: [
      { k: 4, s: 2, out: 8 },
      { k: 3, s: 2, out: 16 },
      { k: 3, s: 1, out: 16 },
    ],
    feat: size > 40 ? 96 : 64,
  };
}

export function visionNetSpec(v: VisionSettings, proprio: number, actions: number, hidden: number[]): VisionNetSpec {
  const e = encoderConvs(v.size);
  return { width: v.size, height: v.size, channels: 3 * v.stack, proprio, actions, convs: e.convs, feat: e.feat, hidden };
}

/** Per-task description of how it is seen. */
export interface VisionTaskSpec {
  /** Which scene the AI camera renders ('reach'). */
  scene: string;
  /** Number of physics-sensor values the environment reports in vision mode. */
  proprioSize: number;
  defaults?: Partial<VisionSettings>;
}

/** First-person (v2) defaults: natural ~90° FOV, small image (see README for the compute trade-off). */
export const FP_VISION_DEFAULTS: Partial<VisionSettings> = { mode: 'fp', size: 48, stack: 1, fov: 90, yawRate: 120, pitchRate: 80 };
