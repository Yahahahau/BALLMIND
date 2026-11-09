import { DEFAULT_DQN_CONFIG, DQNAgent, DQNConfig } from '../rl/dqn';
import type { Learner } from '../rl/learner';
import { VisionDQNAgent } from '../vision/visionAgent';
import type { VisionRig } from '../vision/rig';
import { sanitizeVision, type VisionSettings } from '../vision/config';
import type { Environment, EnvSnapshot, TaskDefinition, TaskNotice } from '../task/types';
import { MODEL_VERSION } from './model';
import { FIXED_DT } from '../physics/physicsWorld';
import type { Counters, EpisodeRecord, ModelBlob, ReplayRecord, StageStats, StageTotals, TrainingStats } from './model';

export type RunState = 'idle' | 'running' | 'paused';

const WINDOW = 50;
const HISTORY_CAP = 4000;
const LOSS_EVERY = 25;
const LOSS_CAP = 4000;
const PERIODIC = 20;
/** Score awarded for a successful episode (plus a speed bonus). */
const SUCCESS_SCORE = 100;

/**
 * Training Manager: owns environment + agent and runs the loop
 *   observe -> act -> N fixed physics steps -> reward -> remember -> learn.
 * It has no DOM dependency, so it runs identically in a Web Worker and in Node tests.
 * Episode resets never touch the network, replay buffer or statistics.
 */
export class TrainingManager {
  readonly task: TaskDefinition;
  readonly env: Environment;
  readonly agent: Learner;
  /** Present for vision tasks: the AI camera + the multimodal agent. */
  readonly vision: { rig: VisionRig; agent: VisionDQNAgent; frame: Uint8Array; renderMs: number } | null = null;
  state: RunState = 'idle';
  speed = 4;
  demo = false;

  private seed: number;
  private obs: Float32Array;
  private nextObs: Float32Array;
  private action = 0;
  private sub = 0;
  private decision = 0;
  private acc = 0;
  private epReward = 0;
  private counters: Counters = { episodes: 0, envSteps: 0, successes: 0, totalScore: 0 };
  private hist: EpisodeRecord[] = [];
  private pending: EpisodeRecord[] = [];
  private pendingTraces: ReplayRecord[] = [];
  private pendingNotices: TaskNotice[] = [];
  private lossHist: number[] = [];
  private pendingLoss: number[] = [];
  private updAcc = 0;
  private stageTotals: StageTotals[] = [];
  private stageBestSeen: number[] = [];
  private demoEpisodes = 0;
  private demoSuccesses = 0;
  private updatesBudget = 0;
  // throughput measurement
  private dpsCount = 0;
  private dpsStart = 0;
  decisionsPerSecond = 0;
  /** Set when anything the UI cares about changed while paused. */
  dirty = true;
  private cachedStats: Partial<TrainingStats> = { avgReward: 0, successRate: 0, avgCompletionTime: 0 };

  private constructor(task: TaskDefinition, env: Environment, agent: Learner, seed: number, rig?: VisionRig) {
    this.task = task;
    this.env = env;
    this.agent = agent;
    if (rig && agent instanceof VisionDQNAgent) {
      const n = 3 * agent.vision.size * agent.vision.size;
      this.vision = { rig, agent, frame: new Uint8Array(n), renderMs: 0 };
    }
    this.seed = seed;
    this.obs = new Float32Array(env.observationSize);
    this.nextObs = new Float32Array(env.observationSize);
    this.beginEpisode();
  }

  /**
   * `vision`: supply an AI-camera rig (and optionally settings) for vision tasks. Without a rig a vision
   * task cannot run – the caller must provide one (GPU in the worker, CPU reference in Node tests).
   */
  static async create(
    task: TaskDefinition,
    seed = 1,
    config: Partial<DQNConfig> = {},
    vision?: { rig: VisionRig; settings?: Partial<VisionSettings> },
  ): Promise<TrainingManager> {
    const env = await task.createEnvironment(seed);
    const cfg = { ...task.dqn, ...config };
    if (task.vision) {
      if (!vision) throw new Error('This task needs the AI vision camera, but no vision renderer is available');
      const settings = sanitizeVision(vision.settings, sanitizeVision(task.vision.defaults));
      vision.rig.configure(settings);
      const agent = new VisionDQNAgent(env.observationSize, env.actionSet.size, settings, cfg, seed);
      return new TrainingManager(task, env, agent, seed, vision.rig);
    }
    return new TrainingManager(task, env, new DQNAgent(env.observationSize, env.actionSet.size, cfg, seed), seed);
  }

  // ---------------- controls ----------------

  start(): void {
    if (this.state !== 'running') this.state = 'running';
    this.dirty = true;
  }
  pause(): void {
    if (this.state === 'running') this.state = 'paused';
    this.dirty = true;
  }
  resume(): void {
    if (this.state === 'paused') this.state = 'running';
    this.dirty = true;
  }
  setSpeed(x: number): void {
    this.speed = x;
  }
  /** Stop: halt and start a fresh (unrecorded) episode. Learning state is kept. */
  stop(): void {
    this.state = 'idle';
    this.beginEpisode();
    this.dirty = true;
  }
  /** Task-specific command (curriculum settings, start checkpoint…). */
  command(cmd: string, arg: unknown): void {
    this.env.command?.(cmd, arg);
    this.dirty = true;
  }

  /** Abort the current episode (not recorded) and start a fresh one. Learning state is untouched. */
  resetEnvironment(): void {
    this.beginEpisode();
    this.dirty = true;
  }

  /** Forget everything the AI has learned. */
  resetAI(): void {
    this.agent.reset(this.seed + Math.floor(Math.random() * 100000));
    this.counters = { episodes: 0, envSteps: 0, successes: 0, totalScore: 0 };
    this.hist = [];
    this.pending = [];
    this.pendingTraces = [];
    this.lossHist = [];
    this.pendingLoss = [];
    this.stageTotals = [];
    this.stageBestSeen = [];
    this.updAcc = 0;
    this.env.command?.('resetProgress', null);
    this.demoEpisodes = 0;
    this.demoSuccesses = 0;
    this.cachedStats = { avgReward: 0, successRate: 0, avgCompletionTime: 0 };
    this.beginEpisode();
    this.dirty = true;
  }

  /** Demo mode: always pick the greedy action, no exploration, no learning, no statistics pollution. */
  setDemo(on: boolean): void {
    this.demo = on;
    this.demoEpisodes = 0;
    this.demoSuccesses = 0;
    this.beginEpisode();
    this.dirty = true;
  }

  // ---------------- main loop ----------------

  private beginEpisode(): void {
    this.obs.set(this.env.reset());
    if (this.vision) {
      this.look();
      this.vision.agent.beginEpisode(this.vision.frame);
    }
    this.sub = 0;
    this.decision = 0;
    this.epReward = 0;
    this.action = this.chooseAction();
  }

  /** Render what the AI camera sees right now into `vision.frame`. */
  private look(): void {
    const v = this.vision!;
    const t = now();
    v.rig.render(this.env.snapshot(), v.frame);
    v.renderMs = v.renderMs === 0 ? now() - t : v.renderMs * 0.95 + (now() - t) * 0.05;
  }

  private chooseAction(): number {
    return this.agent.act(this.obs, !this.demo);
  }

  /**
   * Advance the simulation by `realDt` real seconds × speed, but never use more
   * than `budgetMs` of wall-clock. `maxUpdates` caps gradient updates per call
   * so rendering/UI never starve. Returns decisions made.
   */
  advance(realDt: number, budgetMs: number, maxUpdates = 32): number {
    if (this.state !== 'running') return 0;
    const t0 = now();
    const unlimited = this.speed >= 1000;
    this.acc = unlimited ? Infinity : Math.min(this.acc + realDt * this.speed, 0.25 * this.speed + FIXED_DT * 2);
    this.updatesBudget = maxUpdates;
    let decisions = 0;
    while (this.acc >= FIXED_DT) {
      if (!unlimited) this.acc -= FIXED_DT;
      if (this.substep()) {
        decisions++;
        if (now() - t0 > budgetMs) break;
      } else if (unlimited && this.sub === 0 && now() - t0 > budgetMs) break;
    }
    if (unlimited) this.acc = 0;
    this.measure(decisions);
    this.dirty = true;
    return decisions;
  }

  /** Test helper: run exactly `n` decisions with no time limits. */
  runDecisions(n: number): void {
    this.state = 'running';
    this.updatesBudget = Infinity;
    let done = 0;
    while (done < n) if (this.substep()) done++;
  }

  private measure(d: number): void {
    const t = now();
    if (this.dpsStart === 0) this.dpsStart = t;
    this.dpsCount += d;
    if (t - this.dpsStart >= 1000) {
      this.decisionsPerSecond = (this.dpsCount * 1000) / (t - this.dpsStart);
      this.dpsCount = 0;
      this.dpsStart = t;
    }
  }

  /** One physics step. Returns true when it completed an agent decision. */
  private substep(): boolean {
    this.env.physicsStep(this.action, this.sub === 0);
    this.sub++;
    if (this.sub < this.env.substepsPerDecision) return false;
    this.sub = 0;

    const r = this.env.evaluate(this.decision);
    this.nextObs.set(r.obs);
    if (this.vision) {
      this.look();
      this.vision.agent.pushNext(this.vision.frame);
    }
    this.epReward += r.reward;
    this.decision++;

    if (!this.demo) {
      this.agent.remember(this.obs, this.action, r.reward, this.nextObs, r.terminal);
      this.counters.envSteps++;
      this.updAcc += this.agent.config.updatesPerStep;
      while (this.updAcc >= 1) {
        this.updAcc -= 1;
        if (this.updatesBudget <= 0) {
          this.updAcc = Math.min(this.updAcc, 2);
          break;
        }
        const l = this.agent.train();
        if (l !== null) {
          this.updatesBudget--;
          if (this.agent.trainSteps % LOSS_EVERY === 0) this.pushLoss(this.agent.lastLoss);
        }
      }
    }

    if (r.done) {
      this.finishEpisode(r.success);
      this.beginEpisode();
    } else {
      const t = this.obs;
      this.obs = this.nextObs;
      this.nextObs = t;
      this.vision?.agent.advance();
      this.action = this.chooseAction();
    }
    return true;
  }

  private pushLoss(v: number): void {
    this.lossHist.push(v);
    if (this.lossHist.length > LOSS_CAP) this.lossHist.shift();
    this.pendingLoss.push(v);
  }

  private finishEpisode(success: boolean): void {
    if (this.demo) {
      this.demoEpisodes++;
      if (success) this.demoSuccesses++;
      return;
    }
    const c = this.counters;
    c.episodes++;
    const sum = this.env.episodeSummary?.();
    const partial = !!sum?.partial;
    if (success && !partial) {
      c.successes++;
      c.totalScore += SUCCESS_SCORE + Math.round(50 * (1 - this.decision / this.env.maxDecisions));
    }
    const rec: EpisodeRecord = { ep: c.episodes, reward: this.epReward, success: success && !partial, steps: this.decision };
    if (sum) {
      rec.stage = sum.stage;
      rec.progress = sum.progress;
      rec.checkpoints = sum.checkpoints;
      rec.falls = sum.falls;
      rec.partial = partial;
      this.noteStage(rec, success && !partial);
    }
    this.hist.push(rec);
    if (this.hist.length > HISTORY_CAP) this.hist.shift();
    this.pending.push(rec);
    this.collectTrace(rec, sum !== undefined);
    for (const n of this.env.drainNotices?.() ?? []) {
      this.pendingNotices.push(n);
      if (n.kind === 'unlock') this.agent.boostExploration(0.3);
    }
    this.recomputeWindowStats();
  }

  private noteStage(rec: EpisodeRecord, success: boolean): void {
    const st = rec.stage!;
    while (this.stageTotals.length <= st) this.stageTotals.push({ episodes: 0, successes: 0, falls: 0, bestTime: 0, bestProgress: 0 });
    const t = this.stageTotals[st];
    t.falls += rec.falls ?? 0;
    if (rec.partial) return;
    t.episodes++;
    if (success) {
      t.successes++;
      const time = rec.steps * this.env.substepsPerDecision * FIXED_DT;
      if (t.bestTime === 0 || time < t.bestTime) t.bestTime = time;
    }
    t.bestProgress = Math.max(t.bestProgress, rec.progress ?? 0);
  }

  /** Keep a recording for the replay viewer; the UI decides how many to retain. */
  private collectTrace(rec: EpisodeRecord, multi: boolean): void {
    if (!multi) return;
    const trace = this.env.takeTrace?.();
    if (!trace) return;
    const st = rec.stage!;
    const best = this.stageBestSeen[st] ?? -1;
    const score = (rec.progress ?? 0) + (rec.success ? 1 : 0);
    let kind: ReplayRecord['kind'] = 'recent';
    if (this.stageBestSeen[st] === undefined) kind = 'first';
    else if (score > best + 1e-6) kind = 'milestone';
    else if (rec.ep % PERIODIC === 0) kind = 'periodic';
    this.stageBestSeen[st] = Math.max(best, score);
    this.pendingTraces.push({
      ep: rec.ep,
      stage: st,
      reward: rec.reward,
      success: rec.success,
      progress: rec.progress ?? 0,
      checkpoints: rec.checkpoints ?? 0,
      falls: rec.falls ?? 0,
      kind,
      trace,
    });
  }

  private recomputeWindowStats(): void {
    const recs = this.hist.filter((r) => !r.partial);
    const w = recs.slice(-WINDOW);
    if (w.length === 0) {
      this.cachedStats = { avgReward: 0, successRate: 0, avgCompletionTime: 0 };
      return;
    }
    const dt = this.env.substepsPerDecision * FIXED_DT;
    let rs = 0;
    let ss = 0;
    let ts = 0;
    let tn = 0;
    for (const r of w) {
      rs += r.reward;
      if (r.success) {
        ss++;
        ts += r.steps * dt;
        tn++;
      }
    }
    this.cachedStats = { avgReward: rs / w.length, successRate: ss / w.length, avgCompletionTime: tn ? ts / tn : 0 };
  }

  private stageStats(): StageStats[] {
    const n = Math.max(this.stageTotals.length, (this.task.info.geometry.stages as number | undefined) ?? 0);
    const dt = this.env.substepsPerDecision * FIXED_DT;
    const out: StageStats[] = [];
    for (let i = 0; i < n; i++) {
      const t = this.stageTotals[i] ?? { episodes: 0, successes: 0, falls: 0, bestTime: 0, bestProgress: 0 };
      const w = this.hist.filter((r) => r.stage === i && !r.partial).slice(-WINDOW);
      let rs = 0, ss = 0, ts = 0, tn = 0, ps = 0, cs = 0;
      for (const r of w) {
        rs += r.reward;
        ps += r.progress ?? 0;
        cs += r.checkpoints ?? 0;
        if (r.success) {
          ss++;
          ts += r.steps * dt;
          tn++;
        }
      }
      const m = w.length || 1;
      out.push({
        episodes: t.episodes,
        attempts: w.length,
        successes: t.successes,
        falls: t.falls,
        successRate: w.length ? ss / w.length : 0,
        avgReward: rs / m,
        avgTime: tn ? ts / tn : 0,
        bestTime: t.bestTime,
        avgProgress: ps / m,
        avgCheckpoints: cs / m,
        bestProgress: t.bestProgress,
      });
    }
    return out;
  }

  // ---------------- state out ----------------

  stats(): TrainingStats {
    const s = this.cachedStats;
    return {
      episode: this.counters.episodes + 1,
      episodesDone: this.counters.episodes,
      envSteps: this.counters.envSteps,
      trainSteps: this.agent.trainSteps,
      totalScore: this.counters.totalScore,
      successes: this.counters.successes,
      successRate: s.successRate ?? 0,
      currentReward: this.epReward,
      avgReward: s.avgReward ?? 0,
      avgCompletionTime: s.avgCompletionTime ?? 0,
      epsilon: this.demo ? 0 : this.agent.epsilon,
      loss: this.agent.lastLoss,
      bufferSize: this.agent.replay.size,
      vision: this.vision
        ? {
            size: this.vision.agent.vision.size,
            stack: this.vision.agent.vision.stack,
            params: this.vision.agent.paramCount,
            renderMs: this.vision.renderMs,
            kind: this.vision.rig.kind,
          }
        : undefined,
      demo: this.demo,
      demoEpisodes: this.demoEpisodes,
      demoSuccessRate: this.demoEpisodes ? this.demoSuccesses / this.demoEpisodes : 0,
      extra: this.env.liveStats ? { ...this.env.liveStats(), stages: this.stageStats() } : undefined,
    };
  }

  snapshot(): EnvSnapshot & { action: number } {
    return { ...this.env.snapshot(), action: this.action };
  }

  drainEpisodes(): EpisodeRecord[] {
    const p = this.pending;
    this.pending = [];
    return p;
  }
  drainTraces(): ReplayRecord[] {
    const p = this.pendingTraces;
    this.pendingTraces = [];
    return p;
  }
  drainNotices(): TaskNotice[] {
    const p = this.pendingNotices;
    this.pendingNotices = [];
    return p;
  }
  drainLoss(): number[] {
    const p = this.pendingLoss;
    this.pendingLoss = [];
    return p;
  }

  /** Full history (for the graphs when a model is loaded). */
  fullHistory(): EpisodeRecord[] {
    return this.hist.slice();
  }
  fullLoss(): number[] {
    return this.lossHist.slice();
  }

  // ---------------- persistence ----------------

  /** `withReplay`: include the (int8-quantised) replay buffer – used for IndexedDB saves, not exported files. */
  exportModel(withReplay = false): ModelBlob {
    const a = this.agent.exportState();
    const h = this.hist;
    const multi = h.some((r) => r.stage !== undefined);
    return {
      version: MODEL_VERSION,
      taskId: this.task.info.id,
      obsSize: this.env.observationSize,
      actionCount: this.env.actionSet.size,
      config: { ...this.agent.config },
      online: a.online,
      target: a.target,
      adamM: a.adamM,
      adamV: a.adamV,
      adamT: a.adamT,
      trainSteps: a.trainSteps,
      epsSteps: a.epsSteps,
      counters: { ...this.counters },
      history: {
        reward: Float32Array.from(h, (r) => r.reward),
        success: Uint8Array.from(h, (r) => (r.success ? 1 : 0)),
        steps: Uint16Array.from(h, (r) => r.steps),
        stage: multi ? Uint8Array.from(h, (r) => r.stage ?? 0) : undefined,
        progress: multi ? Float32Array.from(h, (r) => r.progress ?? 0) : undefined,
        checkpoints: multi ? Uint8Array.from(h, (r) => r.checkpoints ?? 0) : undefined,
        falls: multi ? Uint8Array.from(h, (r) => Math.min(255, r.falls ?? 0)) : undefined,
        partial: multi ? Uint8Array.from(h, (r) => (r.partial ? 1 : 0)) : undefined,
      },
      lossHistory: Float32Array.from(this.lossHist),
      stageTotals: this.stageTotals.map((t) => ({ ...t })),
      taskState: this.env.exportState?.(),
      arch: this.agent.arch(),
      replay: withReplay ? this.agent.replay.dump() : undefined,
      savedAt: Date.now(),
    };
  }

  /** Throws a readable error if the blob does not match this task's network shape. */
  importModel(m: ModelBlob): void {
    if (m.taskId !== this.task.info.id) throw new Error(`Model is for task "${m.taskId}", not "${this.task.info.id}"`);
    if (m.obsSize !== this.env.observationSize || m.actionCount !== this.env.actionSet.size)
      throw new Error(
        `Model shape ${m.obsSize}→${m.actionCount} does not match this task (${this.env.observationSize}→${this.env.actionSet.size})`,
      );
    const cfg = { ...DEFAULT_DQN_CONFIG, ...m.config };
    if (m.arch !== undefined ? m.arch !== this.agent.arch() : JSON.stringify(cfg.hidden) !== this.agent.arch())
      throw new Error('Model network architecture does not match (for vision models: image size, frame stack and encoder must match)');
    const n = this.agent.paramCount;
    if (m.online.length !== n || m.target.length !== n || m.adamM.length !== n || m.adamV.length !== n)
      throw new Error('Model weight count does not match this network');
    this.agent.config = cfg;
    this.agent.importState(m);
    if (m.replay) this.agent.loadReplay(m.replay);
    this.counters = { ...m.counters };
    const h = m.history;
    this.hist = [];
    for (let i = 0; i < h.reward.length; i++) {
      const r: EpisodeRecord = {
        ep: m.counters.episodes - h.reward.length + i + 1,
        reward: h.reward[i],
        success: !!h.success[i],
        steps: h.steps[i],
      };
      if (h.stage) {
        r.stage = h.stage[i];
        r.progress = h.progress?.[i] ?? 0;
        r.checkpoints = h.checkpoints?.[i] ?? 0;
        r.falls = h.falls?.[i] ?? 0;
        r.partial = !!h.partial?.[i];
      }
      this.hist.push(r);
    }
    this.lossHist = m.lossHistory ? Array.from(m.lossHistory) : [];
    this.pendingLoss = [];
    this.stageTotals = (m.stageTotals ?? []).map((t) => ({ ...t }));
    this.stageBestSeen = [];
    this.pending = [];
    this.pendingTraces = [];
    if (m.taskState !== undefined) this.env.importState?.(m.taskState);
    this.updAcc = 0;
    this.recomputeWindowStats();
    this.demoEpisodes = 0;
    this.demoSuccesses = 0;
    this.beginEpisode();
    this.dirty = true;
  }
}

const now = (): number => (typeof performance !== 'undefined' ? performance.now() : Date.now());
