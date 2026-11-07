import { DEFAULT_BALL } from '../../agent/ballAgent';

export const OBBY = {
  ball: { ...DEFAULT_BALL, jumpSpeed: 5.0, restitution: 0.05 },
  substepsPerDecision: 6,
  maxFalls: 6,
  /** Decisions without a new best progress before the run is abandoned. */
  stallLimit: 350,
  /** Respawn height above the platform surface (ball centre). */
  respawnLift: 0.08,
  probeDirs: 8,
  probeDist: [1.3, 2.8, 4.8],
  rayRange: 5,
  hazardSlots: 2,
};

export const REWARD = {
  /** Per metre of path progress (potential difference – loops sum to zero). */
  progress: 1.0,
  landing: 1.5,
  checkpoint: 6,
  finish: 50,
  fall: -6,
  timePenalty: 0.004,
};
