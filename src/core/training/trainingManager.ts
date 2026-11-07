import { DEFAULT_DQN_CONFIG, DQNAgent, DQNConfig } from '../rl/dqn';
import type { Environment, EnvSnapshot, TaskDefinition } from '../task/types';
import { FIXED_DT } from '../physics/physicsWorld';
import type { Counters, EpisodeRecord, ModelBlob, TrainingStats } from './model';

export type RunState = 'idle' | 'running' | 'paused';

const WINDOW = 50;
const HISTORY_CAP = 4000;
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
  readonly agent: DQNAgent;
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
  private rewards: number[] = [];
  private successes: number[] = [];
  private steps: number[] = [];
  private pending: EpisodeRecord[] = [];
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

  private constructor(task: TaskDefinition, env: Environment, agent: DQNAgent, seed: number) {
    this.task = task;
    this.env = env;
    this.agent = agent;
    this.seed = seed;
    this.obs = new Float32Array(env.observationSize);
    this.nextObs = new Float32Array(env.observationSize);
    this.beginEpisode();
  }

  static async create(task: TaskDefinition, seed = 1, config: Partial<DQNConfig> = {}): Promise<TrainingManager> {
    const env = await task.createEnvironment(seed);
    const agent = new DQNAgent(env.observationSize, env.actionSet.size, config, seed);
    return new TrainingManager(task, env, agent, seed);
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

  /** Abort the current episode (not recorded) and start a fresh one. Learning state is untouched. */
  resetEnvironment(): void {
    this.beginEpisode();
    this.dirty = true;
  }

  /** Forget everything the AI has learned. */
  resetAI(): void {
    this.agent.reset(this.seed + Math.floor(Math.random() * 100000));
    this.counters = { episodes: 0, envSteps: 0, successes: 0, totalScore: 0 };
    this.rewards = [];
    this.successes = [];
    this.steps = [];
    this.pending = [];
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
    this.sub = 0;
    this.decision = 0;
    this.epReward = 0;
    this.action = this.chooseAction();
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
    this.epReward += r.reward;
    this.decision++;

    if (!this.demo) {
      this.agent.remember(this.obs, this.action, r.reward, this.nextObs, r.terminal);
      this.counters.envSteps++;
      for (let u = 0; u < this.agent.config.updatesPerStep; u++) {
        if (this.updatesBudget <= 0) break;
        if (this.agent.train() !== null) this.updatesBudget--;
      }
    }

    if (r.done) {
      this.finishEpisode(r.success);
      this.beginEpisode();
    } else {
      const t = this.obs;
      this.obs = this.nextObs;
      this.nextObs = t;
      this.action = this.chooseAction();
    }
    return true;
  }

  private finishEpisode(success: boolean): void {
    if (this.demo) {
      this.demoEpisodes++;
      if (success) this.demoSuccesses++;
      return;
    }
    const c = this.counters;
    c.episodes++;
    if (success) {
      c.successes++;
      c.totalScore += SUCCESS_SCORE + Math.round(50 * (1 - this.decision / this.env.maxDecisions));
    }
    this.rewards.push(this.epReward);
    this.successes.push(success ? 1 : 0);
    this.steps.push(this.decision);
    if (this.rewards.length > HISTORY_CAP) {
      this.rewards.shift();
      this.successes.shift();
      this.steps.shift();
    }
    this.pending.push({ ep: c.episodes, reward: this.epReward, success, steps: this.decision });
    this.recomputeWindowStats();
  }

  private recomputeWindowStats(): void {
    const n = Math.min(WINDOW, this.rewards.length);
    if (n === 0) {
      this.cachedStats = { avgReward: 0, successRate: 0, avgCompletionTime: 0 };
      return;
    }
    let rs = 0;
    let ss = 0;
    let ts = 0;
    let tn = 0;
    const dt = (this.env.substepsPerDecision * FIXED_DT) as number;
    for (let i = this.rewards.length - n; i < this.rewards.length; i++) {
      rs += this.rewards[i];
      ss += this.successes[i];
      if (this.successes[i]) {
        ts += this.steps[i] * dt;
        tn++;
      }
    }
    this.cachedStats = { avgReward: rs / n, successRate: ss / n, avgCompletionTime: tn ? ts / tn : 0 };
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
      demo: this.demo,
      demoEpisodes: this.demoEpisodes,
      demoSuccessRate: this.demoEpisodes ? this.demoSuccesses / this.demoEpisodes : 0,
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

  /** Full history (for the graph when a model is loaded). */
  fullHistory(): EpisodeRecord[] {
    const out: EpisodeRecord[] = [];
    const first = this.counters.episodes - this.rewards.length;
    for (let i = 0; i < this.rewards.length; i++)
      out.push({ ep: first + i + 1, reward: this.rewards[i], success: !!this.successes[i], steps: this.steps[i] });
    return out;
  }

  // ---------------- persistence ----------------

  exportModel(): ModelBlob {
    const a = this.agent.exportState();
    return {
      version: 1,
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
        reward: Float32Array.from(this.rewards),
        success: Uint8Array.from(this.successes),
        steps: Uint16Array.from(this.steps),
      },
      savedAt: Date.now(),
    };
  }

  /** Throws if the blob does not match this task's network shape. */
  importModel(m: ModelBlob): void {
    if (m.taskId !== this.task.info.id) throw new Error(`Model is for task "${m.taskId}"`);
    if (m.obsSize !== this.env.observationSize || m.actionCount !== this.env.actionSet.size)
      throw new Error('Model input/output size does not match this task');
    const cfg = { ...DEFAULT_DQN_CONFIG, ...m.config };
    if (JSON.stringify(cfg.hidden) !== JSON.stringify(this.agent.config.hidden))
      throw new Error('Model network architecture does not match');
    this.agent.config = cfg;
    this.agent.importState(m);
    this.counters = { ...m.counters };
    this.rewards = Array.from(m.history.reward);
    this.successes = Array.from(m.history.success);
    this.steps = Array.from(m.history.steps);
    this.pending = [];
    this.recomputeWindowStats();
    this.demoEpisodes = 0;
    this.demoSuccesses = 0;
    this.beginEpisode();
    this.dirty = true;
  }
}

const now = (): number => (typeof performance !== 'undefined' ? performance.now() : Date.now());
