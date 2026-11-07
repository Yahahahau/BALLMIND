export const REWARD = {
  /** Per metre of progress toward the target (potential difference – cannot be farmed by looping). */
  progress: 1.0,
  timePenalty: 0.01,
  jumpPenalty: 0.03,
  success: 10,
  fall: -5,
};

export interface RewardInput {
  prevDist: number;
  dist: number;
  success: boolean;
  fell: boolean;
  jumped: boolean;
}

/**
 * Reward System. Shaping is the plain decrease of distance, so any closed loop
 * sums to zero: the only way to collect reward is to genuinely approach the
 * target (+ the success bonus). Time and unnecessary jumps cost a little;
 * falling off the arena is penalised.
 */
export function computeReward(i: RewardInput): number {
  let r = (i.prevDist - i.dist) * REWARD.progress - REWARD.timePenalty;
  if (i.jumped) r -= REWARD.jumpPenalty;
  if (i.success) r += REWARD.success;
  if (i.fell) r += REWARD.fall;
  return r;
}
