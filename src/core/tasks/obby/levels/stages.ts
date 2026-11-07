import type { StageDef } from '../types';
import { CourseBuilder } from './builder';

const PI = Math.PI;

/** STAGE 01 — THE GAP (Beginner): flat jumps, height changes, steering before a jump. */
function stage1(): StageDef {
  const b = new CourseBuilder().start(7, 6);
  b.plat({ len: 5, w: 5, gap: 1.2 })
    .plat({ len: 4.5, w: 4.5, gap: 1.8 })
    .plat({ len: 4.5, w: 4.5, gap: 2.2, dy: 0.4 })
    .checkpoint({ len: 5, w: 5, gap: 1.5 })
    .plat({ len: 4.5, w: 4.5, gap: 2.6, dy: -0.6 })
    .plat({ len: 4.5, w: 4.5, gap: 2.0, turn: 25 })
    .plat({ len: 4.5, w: 4.5, gap: 2.4, turn: -25, dy: 0.4 })
    .checkpoint({ len: 5, w: 5, gap: 1.8 })
    .plat({ len: 4, w: 4, gap: 2.8 })
    .plat({ len: 4, w: 4, gap: 2.4, dy: 0.6, turn: 15 })
    .plat({ len: 4, w: 4, gap: 3.0, dy: -0.4, turn: -15 })
    .finish({ len: 7, w: 6, gap: 2.0 });
  return {
    id: 1,
    name: 'THE GAP',
    difficulty: 'Beginner',
    blurb: 'Flat platforms, gaps of different lengths and heights. Learn to run up, jump and land.',
    ...b.build(),
    maxDecisions: 700,
    parTime: 45,
  };
}

/** STAGE 02 — NARROW PATH (Hard): precision, turns, slopes, tiny platforms and no run-up room. */
function stage2(): StageDef {
  const b = new CourseBuilder().start(5, 4);
  b.plat({ len: 6, w: 1.8 })
    .plat({ len: 3, w: 1.8, turn: 90 })
    .plat({ len: 4, w: 1.8 })
    .plat({ len: 2, w: 2, gap: 1.6 })
    .checkpoint({ len: 3, w: 2.6, gap: 1.4 })
    .plat({ len: 5, w: 1.7, turn: -90 })
    .plat({ len: 3.5, w: 2.4, tilt: [0, 9] })
    .plat({ len: 2, w: 1.8, gap: 1.8, dy: 0.4 })
    .plat({ len: 2, w: 1.8, gap: 2.0, turn: 45 })
    .plat({ len: 2, w: 1.6, gap: 2.0, turn: -45, dy: -0.3 })
    .checkpoint({ len: 3, w: 2.6, gap: 1.6 })
    .plat({ len: 6, w: 1.8 })
    .plat({ len: 3, w: 1.7, turn: 90 })
    .plat({ len: 4, w: 1.6, turn: -90 })
    .plat({ len: 4, w: 2, tilt: [-8, 0] })
    .plat({ len: 1.8, w: 1.8, gap: 2.2 })
    .plat({ len: 1.8, w: 1.8, gap: 2.4, dy: 0.5 })
    .finish({ len: 5, w: 4, gap: 2.0 });
  return {
    id: 2,
    name: 'NARROW PATH',
    difficulty: 'Hard',
    blurb: 'A path barely wider than the ball: turns, slopes, small platforms and short, precise jumps.',
    ...b.build(),
    maxDecisions: 900,
    parTime: 60,
  };
}

/** STAGE 03 — MOVING PLATFORMS (Very Hard): timing, riding, hopping between moving platforms. */
function stage3(): StageDef {
  const b = new CourseBuilder().start(6, 5);
  b.mover({ axis: 'forward', amp: 2.5, period: 6, len: 4, w: 4, gap: 1.2 })
    .plat({ len: 5, w: 4, gap: 1.2 })
    .mover({ axis: 'up', amp: 1.0, period: 5, len: 4, w: 4, gap: 1.5 })
    .plat({ len: 5, w: 4, gap: 1.5 })
    .checkpoint({ len: 5, w: 5, gap: 0 })
    .mover({ axis: 'right', amp: 3, period: 5.5, len: 3.5, w: 3.5, gap: 2.2 })
    .plat({ len: 5, w: 5, gap: 2.2 })
    .mover({ axis: 'forward', amp: 3, period: 5, len: 3.5, w: 3.5, gap: 1.0, phase: 0 })
    .mover({ axis: 'forward', amp: 3, period: 5, len: 3.5, w: 3.5, gap: 1.0, phase: PI })
    .checkpoint({ len: 5, w: 5, gap: 1.0 })
    .mover({ axis: 'right', amp: 2.5, period: 4.5, len: 3, w: 3, gap: 2.0 })
    .mover({ axis: 'up', amp: 1.0, period: 4, len: 3, w: 3, gap: 1.6, phase: 1 })
    .mover({ axis: 'forward', amp: 2.5, period: 5.5, len: 3, w: 3, gap: 1.2, phase: 1.5 })
    .finish({ len: 7, w: 6, gap: 1.5 });
  return {
    id: 3,
    name: 'MOVING PLATFORMS',
    difficulty: 'Very Hard',
    blurb: 'Platforms slide forward, back, sideways and up and down. Read their position and speed, then time the jump.',
    ...b.build(),
    maxDecisions: 1000,
    parTime: 70,
  };
}

/** STAGE 04 — ROTATION & PRECISION (Extreme). */
function stage4(): StageDef {
  const b = new CourseBuilder().start(6, 5);
  b.spinning({ len: 5, w: 5, speed: 0.6, gap: 1.6 })
    .plat({ len: 4, w: 3, gap: 1.8 })
    .plat({ len: 6, w: 4, gap: 0 })
    .spinner({ length: 3.6, speed: 1.5, height: 0.5 })
    .checkpoint({ len: 4, w: 4, gap: 1.5 })
    .slippery({ len: 5, w: 2.2, gap: 1.2 })
    .slippery({ len: 4, w: 2.2, turn: 90 })
    .plat({ len: 5, w: 3, gap: 1.6 })
    .slider({ width: 2, amp: 1.2, period: 3.2 })
    .plat({ len: 5, w: 3 })
    .slider({ width: 2, amp: 1.2, period: 3.2, phase: PI / 2 })
    .checkpoint({ len: 4, w: 4, gap: 1.0 })
    .spinning({ len: 4, w: 4, speed: -0.9, gap: 1.8 })
    .spinning({ len: 4, w: 4, speed: 0.9, gap: 1.5 })
    .plat({ len: 7, w: 3.2, gap: 1.5 })
    .spinner({ length: 3, speed: -1.9, height: 0.5 })
    .slippery({ len: 3, w: 2, gap: 2.2 })
    .finish({ len: 6, w: 5, gap: 2.0 });
  return {
    id: 4,
    name: 'ROTATION & PRECISION',
    difficulty: 'Extreme',
    blurb: 'Spinning platforms, sweeping bars, rhythmic walls and ice. Momentum, braking and timing decide everything.',
    ...b.build(),
    maxDecisions: 1100,
    parTime: 80,
  };
}

/** STAGE 05 — THE FINAL GAUNTLET (Nightmare): everything combined, almost no resting places. */
function stage5(): StageDef {
  const b = new CourseBuilder().start(8, 5);
  b.plat({ len: 3, w: 3, gap: 4.0 }) // big gap, needs a full run-up
    .plat({ len: 2, w: 1.8, gap: 2.4, dy: 0.4 })
    .plat({ len: 5, w: 1.5, gap: 1.8, turn: 35 })
    .plat({ len: 4, w: 1.5, turn: -70 })
    .checkpoint({ len: 3.2, w: 3, gap: 1.6, turn: 35 })
    .mover({ axis: 'forward', amp: 2.5, period: 5, len: 3, w: 2.6, gap: 1.0 })
    .spinning({ len: 3.6, w: 3.6, speed: 0.8, gap: 1.0 })
    .slippery({ len: 4.5, w: 2, gap: 1.8 })
    .slider({ width: 1.6, amp: 0.9, period: 3, at: 0.3 })
    .plat({ len: 2.2, w: 2.2, gap: 3.2, dy: 0.6 })
    .plat({ len: 2.2, w: 2.2, gap: 3.4, dy: -0.4 })
    .plat({ len: 2.2, w: 2.2, gap: 3.6, dy: 0.5, turn: 25 })
    .checkpoint({ len: 3.2, w: 3, gap: 2.0, turn: -25 })
    .mover({ axis: 'up', amp: 1.0, period: 4, len: 2.6, w: 2.6, gap: 1.4 })
    .mover({ axis: 'right', amp: 2.2, period: 4.2, len: 2.6, w: 2.6, gap: 2.0 })
    .plat({ len: 6.5, w: 2.4, gap: 1.6 })
    .spinner({ length: 2.6, speed: 2.1, height: 0.5 })
    .slippery({ len: 3.5, w: 1.8, gap: 1.4, tilt: [0, 6] })
    .spinning({ len: 3, w: 3, speed: -1.1, gap: 1.8 })
    .plat({ len: 1.8, w: 1.8, gap: 3.8, dy: 0.3 })
    .mover({ axis: 'forward', amp: 2.2, period: 4.6, len: 2.6, w: 2.4, gap: 1.2, phase: 2 })
    .finish({ len: 6, w: 5, gap: 1.2 });
  return {
    id: 5,
    name: 'THE FINAL GAUNTLET',
    difficulty: 'Nightmare',
    blurb: 'Every obstacle at once: huge gaps, tiny winding paths, moving and spinning platforms, ice and sweeping hazards.',
    ...b.build(),
    maxDecisions: 1500,
    parTime: 110,
  };
}

export const STAGE_DEFS: StageDef[] = [stage1(), stage2(), stage3(), stage4(), stage5()];
