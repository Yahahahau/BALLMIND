import { platformTop } from '../kinematics';
import type { CompiledStage, StageDef } from '../types';
import { STAGE_DEFS } from './stages';

export function compileStage(def: StageDef): CompiledStage {
  const cum: number[] = [0];
  const min: [number, number, number] = [Infinity, Infinity, Infinity];
  const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
  let lowTop = Infinity;
  def.platforms.forEach((p, i) => {
    if (i > 0) {
      const a = def.platforms[i - 1].pos;
      cum.push(cum[i - 1] + Math.hypot(p.pos[0] - a[0], p.pos[2] - a[2], (p.pos[1] - a[1]) * 0.5));
    }
    const reach = Math.max(p.size[0], p.size[1]) / 2 + (p.motion ? p.motion.amp : 0);
    const t = platformTop(p, 0);
    lowTop = Math.min(lowTop, p.pos[1] - (p.motion?.axis[1] ? p.motion.amp : 0));
    for (let k = 0; k < 3; k++) {
      const r = k === 1 ? 0 : reach;
      min[k] = Math.min(min[k], t[k] - r);
      max[k] = Math.max(max[k], t[k] + r + (k === 1 ? 1 : 0));
    }
  });
  const checkpointIdx: number[] = [];
  def.platforms.forEach((p, i) => p.kind === 'checkpoint' && checkpointIdx.push(i));
  return {
    def,
    cum,
    total: cum[cum.length - 1],
    checkpointIdx,
    finishIdx: def.platforms.length - 1,
    voidY: lowTop - 4.5,
    bounds: { min, max },
  };
}

export const STAGES = STAGE_DEFS.map(compileStage);
