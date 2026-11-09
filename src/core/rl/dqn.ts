import { DenseNetwork } from '../nn/dense';
import { ReplayBuffer, ReplayDump } from './replayBuffer';
import { Rng } from '../math/rng';
import type { Learner } from './learner';

export interface DQNConfig {
  hidden: number[];
  gamma: number;
  learningRate: number;
  batchSize: number;
  bufferCapacity: number;
  /** Minimum transitions in the buffer before learning starts. */
  minReplay: number;
  /** Hard-copy online -> target every N gradient updates. */
  targetSyncEvery: number;
  epsStart: number;
  epsEnd: number;
  /** Environment steps over which epsilon decays linearly. */
  epsDecaySteps: number;
  gradClip: number;
  huberDelta: number;
  /** Gradient updates per environment decision. */
  updatesPerStep: number;
}

export const DEFAULT_DQN_CONFIG: DQNConfig = {
  hidden: [64, 64],
  gamma: 0.97,
  learningRate: 0.0003,
  batchSize: 32,
  bufferCapacity: 50000,
  minReplay: 1000,
  targetSyncEvery: 1000,
  epsStart: 1.0,
  epsEnd: 0.08,
  epsDecaySteps: 8000,
  gradClip: 10,
  huberDelta: 1.0,
  updatesPerStep: 1,
};

export interface AgentState {
  online: Float32Array;
  target: Float32Array;
  adamM: Float32Array;
  adamV: Float32Array;
  adamT: number;
  trainSteps: number;
  epsSteps: number;
}

/**
 * Double DQN with experience replay, epsilon-greedy exploration, a periodically
 * synchronised target network and an Adam optimiser.
 */
export class DQNAgent implements Learner {
  readonly obsDim: number;
  readonly actionCount: number;
  config: DQNConfig;
  online: DenseNetwork;
  target: DenseNetwork;
  replay: ReplayBuffer;
  trainSteps = 0;
  /** Environment decisions taken while learning – drives the epsilon schedule. */
  epsSteps = 0;
  lastLoss = 0;

  private rng: Rng;
  private adamM: Float32Array;
  private adamV: Float32Array;
  private adamT = 0;
  private grad: Float32Array;
  private readonly q: Float32Array;
  // scratch batch buffers
  private bObs: Float32Array;
  private bNext: Float32Array;
  private bAct: Int32Array;
  private bRew: Float32Array;
  private bDone: Float32Array;
  private bNextAct: Int32Array;
  private dOut: Float32Array;

  constructor(obsDim: number, actionCount: number, config: Partial<DQNConfig> = {}, seed = 1) {
    this.obsDim = obsDim;
    this.actionCount = actionCount;
    this.config = { ...DEFAULT_DQN_CONFIG, ...config };
    this.rng = new Rng(seed * 7919 + 13);
    const sizes = [obsDim, ...this.config.hidden, actionCount];
    this.online = new DenseNetwork(sizes);
    this.target = new DenseNetwork(sizes);
    this.online.init(this.rng);
    this.target.copyFrom(this.online);
    this.replay = new ReplayBuffer(this.config.bufferCapacity, obsDim);
    const n = this.online.params.length;
    this.adamM = new Float32Array(n);
    this.adamV = new Float32Array(n);
    this.grad = new Float32Array(n);
    this.q = new Float32Array(actionCount);
    const B = this.config.batchSize;
    this.bObs = new Float32Array(B * obsDim);
    this.bNext = new Float32Array(B * obsDim);
    this.bAct = new Int32Array(B);
    this.bRew = new Float32Array(B);
    this.bDone = new Float32Array(B);
    this.bNextAct = new Int32Array(B);
    this.dOut = new Float32Array(B * actionCount);
  }

  get paramCount(): number {
    return this.online.params.length;
  }

  arch(): string {
    return JSON.stringify(this.config.hidden);
  }

  get epsilon(): number {
    const c = this.config;
    const f = Math.min(1, this.epsSteps / c.epsDecaySteps);
    return c.epsStart + (c.epsEnd - c.epsStart) * f;
  }

  /** Raise epsilon to at least `eps` (it then decays again) – used when a harder stage unlocks. */
  boostExploration(eps: number): void {
    const c = this.config;
    if (this.epsilon >= eps) return;
    const f = (c.epsStart - eps) / (c.epsStart - c.epsEnd);
    this.epsSteps = Math.floor(Math.max(0, f) * c.epsDecaySteps);
  }

  /** Re-initialise everything (weights, optimiser, replay, schedules). */
  reset(seed?: number): void {
    if (seed !== undefined) this.rng = new Rng(seed * 7919 + 13);
    this.online.init(this.rng);
    this.target.copyFrom(this.online);
    this.replay.clear();
    this.adamM.fill(0);
    this.adamV.fill(0);
    this.adamT = 0;
    this.trainSteps = 0;
    this.epsSteps = 0;
    this.lastLoss = 0;
  }

  greedyAction(obs: Float32Array): number {
    this.online.predict(obs, this.q);
    let best = 0;
    for (let a = 1; a < this.actionCount; a++) if (this.q[a] > this.q[best]) best = a;
    return best;
  }

  /** Epsilon-greedy action selection. */
  act(obs: Float32Array, explore: boolean): number {
    if (explore && this.rng.next() < this.epsilon) return this.rng.int(this.actionCount);
    return this.greedyAction(obs);
  }

  qValues(obs: Float32Array): Float32Array {
    this.online.predict(obs, this.q);
    return this.q;
  }

  remember(obs: Float32Array, action: number, reward: number, next: Float32Array, terminal: boolean): void {
    this.replay.add(obs, action, reward, next, terminal);
    this.epsSteps++;
  }

  get canTrain(): boolean {
    return this.replay.size >= Math.max(this.config.minReplay, this.config.batchSize);
  }

  /** One gradient update on a replay mini-batch. Returns the loss, or null if not enough data. */
  train(): number | null {
    if (!this.canTrain) return null;
    const c = this.config;
    const B = c.batchSize;
    const A = this.actionCount;
    this.replay.sample(B, this.rng, this.bObs, this.bAct, this.bRew, this.bNext, this.bDone);

    // Double DQN: action chosen by the online net, evaluated by the target net.
    const qNextOnline = this.online.forwardBatch(this.bNext, B);
    for (let b = 0; b < B; b++) {
      let best = 0;
      const o = b * A;
      for (let a = 1; a < A; a++) if (qNextOnline[o + a] > qNextOnline[o + best]) best = a;
      this.bNextAct[b] = best;
    }
    const qNextTarget = this.target.forwardBatch(this.bNext, B);

    // Forward on the current states last so the online net caches the right activations.
    const q = this.online.forwardBatch(this.bObs, B);
    this.dOut.fill(0);
    let loss = 0;
    for (let b = 0; b < B; b++) {
      const o = b * A;
      const y = this.bRew[b] + c.gamma * (1 - this.bDone[b]) * qNextTarget[o + this.bNextAct[b]];
      const err = q[o + this.bAct[b]] - y;
      const ae = Math.abs(err);
      if (ae <= c.huberDelta) {
        loss += 0.5 * err * err;
        this.dOut[o + this.bAct[b]] = err / B;
      } else {
        loss += c.huberDelta * (ae - 0.5 * c.huberDelta);
        this.dOut[o + this.bAct[b]] = (Math.sign(err) * c.huberDelta) / B;
      }
    }
    loss /= B;

    this.online.backward(this.dOut, B, this.grad);

    // Global gradient-norm clipping
    let norm = 0;
    for (let i = 0; i < this.grad.length; i++) norm += this.grad[i] * this.grad[i];
    norm = Math.sqrt(norm);
    const scale = norm > c.gradClip ? c.gradClip / norm : 1;

    // Adam
    const b1 = 0.9;
    const b2 = 0.999;
    this.adamT++;
    const bc1 = 1 - Math.pow(b1, this.adamT);
    const bc2 = 1 - Math.pow(b2, this.adamT);
    const p = this.online.params;
    for (let i = 0; i < p.length; i++) {
      const g = this.grad[i] * scale;
      const m = (this.adamM[i] = b1 * this.adamM[i] + (1 - b1) * g);
      const v = (this.adamV[i] = b2 * this.adamV[i] + (1 - b2) * g * g);
      p[i] -= (c.learningRate * (m / bc1)) / (Math.sqrt(v / bc2) + 1e-8);
    }

    this.trainSteps++;
    if (this.trainSteps % c.targetSyncEvery === 0) this.target.copyFrom(this.online);
    this.lastLoss = this.lastLoss === 0 ? loss : this.lastLoss * 0.98 + loss * 0.02;
    return loss;
  }

  exportState(): AgentState {
    return {
      online: this.online.params.slice(),
      target: this.target.params.slice(),
      adamM: this.adamM.slice(),
      adamV: this.adamV.slice(),
      adamT: this.adamT,
      trainSteps: this.trainSteps,
      epsSteps: this.epsSteps,
    };
  }

  importState(s: AgentState): void {
    this.online.params.set(s.online);
    this.target.params.set(s.target);
    this.adamM.set(s.adamM);
    this.adamV.set(s.adamV);
    this.adamT = s.adamT;
    this.trainSteps = s.trainSteps;
    this.epsSteps = s.epsSteps;
    this.replay.clear();
  }

  /** Restore replay memory saved by `ReplayBuffer.dump`. */
  loadReplay(d: ReplayDump): void {
    this.replay.load(d);
  }
}
