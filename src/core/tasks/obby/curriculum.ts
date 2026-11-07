import { Rng } from '../../math/rng';

export interface CurriculumConfig {
  /** Auto: train the frontier stage, rehearse earlier ones. Manual: only `manualStage`. */
  auto: boolean;
  manualStage: number;
  /** Minimum success rate over the window to unlock the next stage. */
  threshold: number;
  /** Number of recent full-run episodes the criteria are measured on. */
  window: number;
  /** Minimum average path progress over the window. */
  minProgress: number;
  /** Average completion time of successful runs must be <= parTime × this (0 = ignore). */
  timeFactor: number;
  /** Share of auto-mode episodes spent on already-unlocked earlier stages (anti-forgetting). */
  rehearsal: number;
}

export const DEFAULT_CURRICULUM: CurriculumConfig = {
  auto: true,
  manualStage: 0,
  threshold: 0.6,
  window: 20,
  minProgress: 0.7,
  timeFactor: 3,
  rehearsal: 0.25,
};

interface Result {
  success: boolean;
  progress: number;
  time: number;
}

export interface CurriculumState {
  config: CurriculumConfig;
  unlocked: number;
  results: Result[][];
}

/** Stage Manager: which stage to train next and when the next one unlocks (measured on real results). */
export class Curriculum {
  config: CurriculumConfig = { ...DEFAULT_CURRICULUM };
  unlocked = 0;
  private results: Result[][];

  constructor(private readonly stageCount: number, private readonly parTimes: number[]) {
    this.results = Array.from({ length: stageCount }, () => []);
  }

  /** Forget unlock progress and results (user-chosen settings are kept). */
  resetProgress(): void {
    this.unlocked = 0;
    this.results = Array.from({ length: this.stageCount }, () => []);
  }

  chooseStage(rng: Rng): number {
    if (!this.config.auto) return Math.min(this.stageCount - 1, Math.max(0, this.config.manualStage));
    if (this.unlocked > 0 && rng.next() < this.config.rehearsal) return rng.int(this.unlocked);
    return this.unlocked;
  }

  /** Records a finished full-run episode. Returns the newly unlocked stage index, if any. */
  record(stage: number, r: Result): number | null {
    const list = this.results[stage];
    list.push(r);
    if (list.length > 100) list.shift();
    if (!this.config.auto || stage !== this.unlocked || stage >= this.stageCount - 1) return null;
    if (this.criteriaMet(stage)) {
      this.unlocked = stage + 1;
      return this.unlocked;
    }
    return null;
  }

  criteriaMet(stage: number): boolean {
    const c = this.config;
    const w = this.results[stage].slice(-c.window);
    if (w.length < c.window) return false;
    const ok = w.filter((x) => x.success);
    const rate = ok.length / w.length;
    const prog = w.reduce((a, x) => a + x.progress, 0) / w.length;
    const time = ok.length ? ok.reduce((a, x) => a + x.time, 0) / ok.length : Infinity;
    return rate >= c.threshold && prog >= c.minProgress && (c.timeFactor <= 0 || time <= this.parTimes[stage] * c.timeFactor);
  }

  /** Progress (0..1) toward unlocking the next stage, for the dashboard. */
  unlockProgress(stage: number): number {
    const c = this.config;
    const w = this.results[stage].slice(-c.window);
    if (!w.length) return 0;
    const rate = w.filter((x) => x.success).length / w.length;
    return Math.min(1, (rate / c.threshold) * Math.min(1, w.length / c.window));
  }

  exportState(): CurriculumState {
    return { config: { ...this.config }, unlocked: this.unlocked, results: this.results.map((r) => r.slice()) };
  }

  importState(s: CurriculumState): void {
    this.config = { ...DEFAULT_CURRICULUM, ...s.config };
    this.unlocked = Math.min(this.stageCount - 1, s.unlocked ?? 0);
    this.results = Array.from({ length: this.stageCount }, (_, i) => (s.results?.[i] ?? []).slice(-100));
  }
}
