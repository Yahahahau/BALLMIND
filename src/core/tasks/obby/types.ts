import type { Vec3 } from '../../math/quat';

/** pos(t) = base + axis * amp * sin(2π t / period + phase) */
export interface MotionDef {
  axis: Vec3;
  amp: number;
  period: number;
  phase: number;
}
/** yaw(t) = yaw0 + speed * t (rad/s) */
export interface SpinDef {
  speed: number;
  phase: number;
}

export type PlatformKind = 'start' | 'normal' | 'checkpoint' | 'finish' | 'moving' | 'rotating' | 'slippery';

export interface PlatformDef {
  kind: PlatformKind;
  /** Centre of the TOP surface at t = 0 (before motion). */
  pos: Vec3;
  /** [width (local x), length (local z)] */
  size: [number, number];
  yaw: number;
  /** [pitch about x, roll about z] in radians, applied in the platform's local frame. */
  tilt?: [number, number];
  thick: number;
  friction: number;
  motion?: MotionDef;
  spin?: SpinDef;
}

export type HazardKind = 'spinner' | 'slider';
export interface HazardDef {
  kind: HazardKind;
  /** Centre of the bar/wall body. */
  pos: Vec3;
  /** Full size [x, y, z] in local frame (spinner: x = bar length). */
  size: Vec3;
  yaw: number;
  motion?: MotionDef;
  spin?: SpinDef;
}

export interface StageDef {
  id: number;
  name: string;
  difficulty: string;
  blurb: string;
  platforms: PlatformDef[];
  hazards: HazardDef[];
  /** Budget in agent decisions (10 per simulated second) for one run. */
  maxDecisions: number;
  /** A reference "good" completion time in seconds (used by the curriculum speed criterion). */
  parTime: number;
}

export interface CompiledStage {
  def: StageDef;
  /** Cumulative path length at each platform (platform i = waypoint i). */
  cum: number[];
  total: number;
  checkpointIdx: number[];
  finishIdx: number;
  /** Ball falling below this height counts as a fall. */
  voidY: number;
  bounds: { min: Vec3; max: Vec3 };
}
