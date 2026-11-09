import { Rng } from '../math/rng';
import { DEFAULT_DQN_CONFIG, type AgentState, type DQNConfig } from '../rl/dqn';
import type { Learner } from '../rl/learner';
import type { ReplayDump } from '../rl/replayBuffer';
import { VisionNet } from './visionNet';
import { visionNetSpec, type VisionSettings } from './config';

/**
 * Double DQN whose Q-network is the multimodal VisionNet: stacked RGB frames from the
 * AI camera + physics sensors. Replay stores every camera frame ONCE (uint8) in a ring
 * pool; a transition only keeps frame ids, so a stack of K frames costs no extra memory.
 *
 * Usage per episode (driven by the TrainingManager):
 *   beginEpisode(frame) → act(proprio) … then for every decision:
 *   pushNext(frame) → remember(proprio, a, r, nextProprio, terminal) → advance()
 */
export class VisionDQNAgent implements Learner {
  readonly vision: VisionSettings;
  readonly obsDim: number;
  readonly actionCount: number;
  config: DQNConfig;
  online: VisionNet;
  target: VisionNet;
  trainSteps = 0;
  epsSteps = 0;
  lastLoss = 0;
  readonly replay: VisionReplay;

  private rng: Rng;
  private adamM: Float32Array;
  private adamV: Float32Array;
  private adamT = 0;
  private grad: Float32Array;
  private readonly q: Float32Array;
  private readonly frameLen: number;
  private readonly stackLen: number;
  private readonly K: number;
  private cur: Int32Array;
  private nxt: Int32Array;
  private single: Float32Array;
  // batch scratch
  private bFrames: Float32Array;
  private bNextFrames: Float32Array;
  private bObs: Float32Array;
  private bNext: Float32Array;
  private bAct = new Int32Array(0);
  private bRew: Float32Array;
  private bDone: Float32Array;
  private bNextAct: Int32Array;
  private dOut: Float32Array;

  constructor(proprio: number, actionCount: number, vision: VisionSettings, config: Partial<DQNConfig> = {}, seed = 1) {
    this.vision = vision;
    this.obsDim = proprio;
    this.actionCount = actionCount;
    this.config = { ...DEFAULT_DQN_CONFIG, ...config };
    this.rng = new Rng(seed * 7919 + 13);
    const spec = visionNetSpec(vision, proprio, actionCount, this.config.hidden);
    this.online = new VisionNet(spec);
    this.target = new VisionNet(spec);
    this.online.init(this.rng);
    this.target.copyFrom(this.online);
    this.K = vision.stack;
    this.frameLen = 3 * vision.size * vision.size;
    this.stackLen = this.frameLen * this.K;
    this.replay = new VisionReplay(this.config.bufferCapacity, proprio, this.frameLen, this.K);
    const n = this.online.params.length;
    this.adamM = new Float32Array(n);
    this.adamV = new Float32Array(n);
    this.grad = new Float32Array(n);
    this.q = new Float32Array(actionCount);
    this.cur = new Int32Array(this.K);
    this.nxt = new Int32Array(this.K);
    this.single = new Float32Array(this.stackLen);
    const B = this.config.batchSize;
    this.bFrames = new Float32Array(B * this.stackLen);
    this.bNextFrames = new Float32Array(B * this.stackLen);
    this.bObs = new Float32Array(B * proprio);
    this.bNext = new Float32Array(B * proprio);
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
    return JSON.stringify({ hidden: this.config.hidden, size: this.vision.size, stack: this.vision.stack, convs: this.online.spec.convs, feat: this.online.spec.feat });
  }

  get epsilon(): number {
    const c = this.config;
    return c.epsStart + (c.epsEnd - c.epsStart) * Math.min(1, this.epsSteps / c.epsDecaySteps);
  }

  boostExploration(eps: number): void {
    const c = this.config;
    if (this.epsilon >= eps) return;
    const f = (c.epsStart - eps) / (c.epsStart - c.epsEnd);
    this.epsSteps = Math.floor(Math.max(0, f) * c.epsDecaySteps);
  }

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

  // ---------------- perception ----------------

  /** First camera frame of an episode: the stack is filled with copies of it. */
  beginEpisode(frame: Uint8Array): void {
    const id = this.replay.addFrame(frame);
    this.cur.fill(id);
  }

  /** Camera frame observed after the latest decision (becomes the newest frame of the next state). */
  pushNext(frame: Uint8Array): void {
    const id = this.replay.addFrame(frame);
    for (let k = 0; k < this.K - 1; k++) this.nxt[k] = this.cur[k + 1];
    this.nxt[this.K - 1] = id;
  }

  /** The next state becomes the current one (episode continues). */
  advance(): void {
    this.cur.set(this.nxt);
  }

  private gather(ids: Int32Array, out: Float32Array, offset: number): void {
    for (let k = 0; k < this.K; k++) this.replay.readFrame(ids[k], out, offset + k * this.frameLen);
  }

  greedyAction(obs: Float32Array): number {
    this.gather(this.cur, this.single, 0);
    this.online.predict(this.single, obs, this.q);
    let best = 0;
    for (let a = 1; a < this.actionCount; a++) if (this.q[a] > this.q[best]) best = a;
    return best;
  }

  act(obs: Float32Array, explore: boolean): number {
    if (explore && this.rng.next() < this.epsilon) return this.rng.int(this.actionCount);
    return this.greedyAction(obs);
  }

  qValues(obs: Float32Array): Float32Array {
    this.gather(this.cur, this.single, 0);
    this.online.predict(this.single, obs, this.q);
    return this.q;
  }

  remember(obs: Float32Array, action: number, reward: number, next: Float32Array, terminal: boolean): void {
    this.replay.add(obs, action, reward, next, terminal, this.cur, this.nxt);
    this.epsSteps++;
  }

  get canTrain(): boolean {
    return this.replay.size >= Math.max(this.config.minReplay, this.config.batchSize);
  }

  train(): number | null {
    if (!this.canTrain) return null;
    const c = this.config;
    const B = c.batchSize;
    const A = this.actionCount;
    this.replay.sample(B, this.rng, this.bFrames, this.bObs, this.bAct, this.bRew, this.bNextFrames, this.bNext, this.bDone);

    const qNextOnline = this.online.forwardBatch(this.bNextFrames, this.bNext, B);
    for (let b = 0; b < B; b++) {
      let best = 0;
      const o = b * A;
      for (let a = 1; a < A; a++) if (qNextOnline[o + a] > qNextOnline[o + best]) best = a;
      this.bNextAct[b] = best;
    }
    const qNextTarget = this.target.forwardBatch(this.bNextFrames, this.bNext, B);
    const q = this.online.forwardBatch(this.bFrames, this.bObs, B);
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

    let norm = 0;
    for (let i = 0; i < this.grad.length; i++) norm += this.grad[i] * this.grad[i];
    norm = Math.sqrt(norm);
    const scale = norm > c.gradClip ? c.gradClip / norm : 1;
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

  /** Vision replay (raw frames) is large and not persisted – the buffer simply refills. */
  loadReplay(_d: ReplayDump): void {}

  /** Free big buffers. */
  dispose(): void {
    this.replay.clear();
  }
}

/** Transition store + uint8 frame pool. Frames are addressed by monotonically increasing ids. */
export class VisionReplay {
  readonly capacity: number;
  private readonly pool: Uint8Array;
  private readonly poolCap: number;
  private nextFrameId = 0;
  private readonly obs: Float32Array;
  private readonly next: Float32Array;
  private readonly action: Int32Array;
  private readonly reward: Float32Array;
  private readonly done: Float32Array;
  private readonly ids: Int32Array;
  private readonly nids: Int32Array;
  private head = 0;
  private count = 0;

  constructor(
    capacity: number,
    private readonly obsDim: number,
    private readonly frameLen: number,
    private readonly K: number,
  ) {
    this.capacity = capacity;
    // each decision adds one frame, each episode start one more: a little headroom + validity check on sampling
    this.poolCap = capacity + 256;
    this.pool = new Uint8Array(this.poolCap * frameLen);
    this.obs = new Float32Array(capacity * obsDim);
    this.next = new Float32Array(capacity * obsDim);
    this.action = new Int32Array(capacity);
    this.reward = new Float32Array(capacity);
    this.done = new Float32Array(capacity);
    this.ids = new Int32Array(capacity * K);
    this.nids = new Int32Array(capacity * K);
  }

  get size(): number {
    return this.count;
  }

  dump(): ReplayDump | undefined {
    return undefined;
  }

  clear(): void {
    this.head = 0;
    this.count = 0;
  }

  addFrame(frame: Uint8Array): number {
    const id = this.nextFrameId++;
    this.pool.set(frame.subarray(0, this.frameLen), (id % this.poolCap) * this.frameLen);
    return id;
  }

  /** Writes frame `id` (normalised to [0,1]) into out[offset…]. */
  readFrame(id: number, out: Float32Array, offset: number): void {
    const base = (id % this.poolCap) * this.frameLen;
    const pool = this.pool;
    for (let i = 0; i < this.frameLen; i++) out[offset + i] = pool[base + i] * (1 / 255);
  }

  private alive(id: number): boolean {
    return id > this.nextFrameId - this.poolCap;
  }

  add(obs: Float32Array, action: number, reward: number, next: Float32Array, terminal: boolean, cur: Int32Array, nxt: Int32Array): void {
    const d = this.obsDim;
    const i = this.head;
    this.obs.set(obs.subarray(0, d), i * d);
    this.next.set(next.subarray(0, d), i * d);
    this.action[i] = action;
    this.reward[i] = reward;
    this.done[i] = terminal ? 1 : 0;
    this.ids.set(cur, i * this.K);
    this.nids.set(nxt, i * this.K);
    this.head = (i + 1) % this.capacity;
    if (this.count < this.capacity) this.count++;
  }

  sample(
    B: number,
    rng: Rng,
    frames: Float32Array,
    obs: Float32Array,
    action: Int32Array,
    reward: Float32Array,
    nextFrames: Float32Array,
    next: Float32Array,
    done: Float32Array,
  ): void {
    const d = this.obsDim;
    const K = this.K;
    const fl = this.frameLen;
    for (let b = 0; b < B; b++) {
      let i = 0;
      for (let tries = 0; tries < 20; tries++) {
        i = rng.int(this.count);
        if (this.alive(this.ids[i * K])) break; // oldest frame of the stack still in the pool
      }
      obs.set(this.obs.subarray(i * d, i * d + d), b * d);
      next.set(this.next.subarray(i * d, i * d + d), b * d);
      action[b] = this.action[i];
      reward[b] = this.reward[i];
      done[b] = this.done[i];
      for (let k = 0; k < K; k++) {
        this.readFrame(this.ids[i * K + k], frames, (b * K + k) * fl);
        this.readFrame(this.nids[i * K + k], nextFrames, (b * K + k) * fl);
      }
    }
  }
}
