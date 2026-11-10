import { Rng } from '../math/rng';
import { DEFAULT_DQN_CONFIG, type AgentState, type DQNConfig } from '../rl/dqn';
import type { Learner } from '../rl/learner';
import type { ReplayDump } from '../rl/replayBuffer';
import { RecurrentVisionNet, type RecurrentSpec } from './recurrentNet';
import { encoderConvs, type VisionSettings } from './config';

/** Sequence-training profile (R2D2-lite). Zero initial state per sampled sequence + burn-in. */
export interface SeqProfile {
  seqLen: number;
  burnIn: number;
}
export const DEFAULT_SEQ: SeqProfile = { seqLen: 8, burnIn: 3 };

/**
 * Vision Agent V2: CNN encoder + GRU memory + physics sensors → two Q branches (movement / gaze) trained
 * end-to-end by recurrent Double DQN from real Rapier rewards. Actions are composite: a = move·G + gaze, so
 * the existing Learner interface and replay logic keep working, while each branch bootstraps on its own
 * argmax (branching DQN), letting movement and gaze decisions be learned independently.
 */
export class RecurrentVisionAgent implements Learner {
  readonly vision: VisionSettings;
  readonly M: number;
  readonly G: number;
  readonly obsDim: number;
  config: DQNConfig;
  online: RecurrentVisionNet;
  target: RecurrentVisionNet;
  trainSteps = 0;
  epsSteps = 0;
  lastLoss = 0;
  readonly replay: SeqReplay;

  private rng: Rng;
  private adamM: Float32Array;
  private adamV: Float32Array;
  private adamT = 0;
  private grad: Float32Array;
  private readonly frameLen: number;
  private curId = 0;
  private nxtId = 0;
  private hidden: Float32Array;
  private readonly q: Float32Array;
  private readonly one: Float32Array;
  private readonly seq: SeqProfile;
  // batch scratch (time-major)
  private bFrames: Float32Array;
  private bProp: Float32Array;
  private bAct: Int32Array;
  private bRew: Float32Array;
  private bDone: Float32Array;
  episodeId = 0;

  constructor(proprio: number, moveActions: number, gazeActions: number, vision: VisionSettings, config: Partial<DQNConfig> = {}, seed = 1, seq: SeqProfile = DEFAULT_SEQ) {
    this.vision = vision;
    this.M = moveActions;
    this.G = gazeActions;
    this.obsDim = proprio;
    this.seq = seq;
    this.config = { ...DEFAULT_DQN_CONFIG, ...config };
    this.rng = new Rng(seed * 7919 + 29);
    const e = encoderConvs(vision.size);
    const spec: RecurrentSpec = {
      width: vision.size,
      height: vision.size,
      channels: 3,
      proprio,
      convs: e.convs,
      feat: e.feat,
      gru: 64,
      hidden: this.config.hidden[0] ?? 64,
      moveActions,
      gazeActions,
    };
    this.online = new RecurrentVisionNet(spec);
    this.target = new RecurrentVisionNet(spec);
    this.online.init(this.rng);
    this.target.copyFrom(this.online);
    this.frameLen = 3 * vision.size * vision.size;
    this.replay = new SeqReplay(this.config.bufferCapacity, proprio, this.frameLen);
    const n = this.online.params.length;
    this.adamM = new Float32Array(n);
    this.adamV = new Float32Array(n);
    this.grad = new Float32Array(n);
    this.hidden = new Float32Array(this.online.H);
    this.q = new Float32Array(moveActions + gazeActions);
    this.one = new Float32Array(this.frameLen);
    const B = this.config.batchSize;
    const T = seq.seqLen + 1;
    this.bFrames = new Float32Array(T * B * this.frameLen);
    this.bProp = new Float32Array(T * B * proprio);
    this.bAct = new Int32Array(seq.seqLen * B);
    this.bRew = new Float32Array(seq.seqLen * B);
    this.bDone = new Float32Array(seq.seqLen * B);
  }

  get actionCount(): number {
    return this.M * this.G;
  }
  get paramCount(): number {
    return this.online.params.length;
  }
  arch(): string {
    const s = this.online.spec;
    return JSON.stringify({ v: 2, size: s.width, convs: s.convs, feat: s.feat, gru: s.gru, hidden: s.hidden, M: this.M, G: this.G });
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
    if (seed !== undefined) this.rng = new Rng(seed * 7919 + 29);
    this.online.init(this.rng);
    this.target.copyFrom(this.online);
    this.replay.clear();
    this.adamM.fill(0);
    this.adamV.fill(0);
    this.adamT = 0;
    this.trainSteps = 0;
    this.epsSteps = 0;
    this.lastLoss = 0;
    this.hidden.fill(0);
  }

  // ---------------- perception / memory ----------------
  beginEpisode(frame: Uint8Array): void {
    this.episodeId++;
    this.curId = this.replay.addFrame(frame);
    this.hidden.fill(0); // the GRU starts every episode with an empty memory
  }
  pushNext(frame: Uint8Array): void {
    this.nxtId = this.replay.addFrame(frame);
  }
  advance(): void {
    this.curId = this.nxtId;
  }

  /** Greedy composite action; advances the GRU state with the current frame. */
  private forwardCurrent(obs: Float32Array): void {
    this.replay.readFrame(this.curId, this.one, 0);
    this.online.predictStep(this.one, obs, this.hidden, this.q);
  }
  private argmax(off: number, n: number): number {
    let best = 0;
    for (let a = 1; a < n; a++) if (this.q[off + a] > this.q[off + best]) best = a;
    return best;
  }

  act(obs: Float32Array, explore: boolean): number {
    this.forwardCurrent(obs); // always: keeps the recurrent state consistent with what training replays
    let m = this.argmax(0, this.M);
    let g = this.argmax(this.M, this.G);
    if (explore) {
      const eps = this.epsilon;
      if (this.rng.next() < eps) m = this.rng.int(this.M);
      // exploratory gaze: usually keep looking where it looks, sometimes try something (index 0 = hold)
      if (this.rng.next() < eps) g = this.rng.next() < 0.5 ? 0 : this.rng.int(this.G);
    }
    return m * this.G + g;
  }
  greedyAction(obs: Float32Array): number {
    return this.act(obs, false);
  }

  remember(obs: Float32Array, action: number, reward: number, next: Float32Array, terminal: boolean): void {
    this.replay.add(obs, action, reward, next, terminal, this.curId, this.nxtId, this.episodeId);
    this.epsSteps++;
  }

  get canTrain(): boolean {
    return this.replay.size >= Math.max(this.config.minReplay, this.config.batchSize * 2);
  }

  train(): number | null {
    if (!this.canTrain) return null;
    const c = this.config;
    const B = c.batchSize;
    const L = this.seq.seqLen;
    const T = L + 1;
    const burn = this.seq.burnIn;
    const M = this.M;
    const G = this.G;
    const A = M + G;
    if (!this.replay.sampleSequences(B, L, this.rng, this.bFrames, this.bProp, this.bAct, this.bRew, this.bDone)) return null;

    const qOn = this.online.unroll(this.bFrames, this.bProp, T, B);
    const qTg = this.target.unroll(this.bFrames, this.bProp, T, B);
    const dQ = new Float32Array(T * B * A);
    let loss = 0;
    let cnt = 0;
    for (let t = burn; t < L; t++) {
      for (let b = 0; b < B; b++) {
        const i = t * B + b;
        const n = (t + 1) * B + b;
        const comp = this.bAct[i];
        const am = Math.floor(comp / G);
        const ag = comp % G;
        const nd = (1 - this.bDone[i]) * c.gamma;
        for (let br = 0; br < 2; br++) {
          const off = br === 0 ? 0 : M;
          const cntA = br === 0 ? M : G;
          const a = br === 0 ? am : ag;
          let best = 0;
          for (let k = 1; k < cntA; k++) if (qOn[n * A + off + k] > qOn[n * A + off + best]) best = k;
          const y = this.bRew[i] + nd * qTg[n * A + off + best];
          const err = qOn[i * A + off + a] - y;
          const ae = Math.abs(err);
          if (ae <= c.huberDelta) {
            loss += 0.5 * err * err;
            dQ[i * A + off + a] = err;
          } else {
            loss += c.huberDelta * (ae - 0.5 * c.huberDelta);
            dQ[i * A + off + a] = Math.sign(err) * c.huberDelta;
          }
          cnt++;
        }
      }
    }
    const inv = 1 / cnt;
    for (let i = 0; i < dQ.length; i++) dQ[i] *= inv;
    loss *= inv;
    this.online.backward(dQ, T, B, burn, this.grad, L);

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
  loadReplay(_d: ReplayDump): void {}
  dispose(): void {
    this.replay.clear();
  }
}

/** Step-ordered replay with a uint8 frame pool; samples contiguous same-episode sequences. */
export class SeqReplay {
  private readonly pool: Uint8Array;
  private readonly poolCap: number;
  private nextFrameId = 0;
  private readonly obs: Float32Array;
  private readonly next: Float32Array;
  private readonly action: Int32Array;
  private readonly reward: Float32Array;
  private readonly done: Float32Array;
  private readonly fo: Int32Array;
  private readonly fn: Int32Array;
  private readonly ep: Int32Array;
  private head = 0;
  private count = 0;

  constructor(private readonly capacity: number, private readonly obsDim: number, private readonly frameLen: number) {
    this.poolCap = capacity * 2 + 64;
    this.pool = new Uint8Array(this.poolCap * frameLen);
    this.obs = new Float32Array(capacity * obsDim);
    this.next = new Float32Array(capacity * obsDim);
    this.action = new Int32Array(capacity);
    this.reward = new Float32Array(capacity);
    this.done = new Float32Array(capacity);
    this.fo = new Int32Array(capacity);
    this.fn = new Int32Array(capacity);
    this.ep = new Int32Array(capacity);
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
  readFrame(id: number, out: Float32Array, off: number): void {
    const base = (id % this.poolCap) * this.frameLen;
    for (let i = 0; i < this.frameLen; i++) out[off + i] = this.pool[base + i] * (1 / 255);
  }
  add(obs: Float32Array, action: number, reward: number, next: Float32Array, terminal: boolean, fo: number, fn: number, ep: number): void {
    const d = this.obsDim;
    const i = this.head;
    this.obs.set(obs.subarray(0, d), i * d);
    this.next.set(next.subarray(0, d), i * d);
    this.action[i] = action;
    this.reward[i] = reward;
    this.done[i] = terminal ? 1 : 0;
    this.fo[i] = fo;
    this.fn[i] = fn;
    this.ep[i] = ep;
    this.head = (i + 1) % this.capacity;
    if (this.count < this.capacity) this.count++;
  }

  /** Time-major output (index t*B+b); frames/proprio have L+1 time steps, the rest L. */
  sampleSequences(B: number, L: number, rng: Rng, frames: Float32Array, prop: Float32Array, act: Int32Array, rew: Float32Array, done: Float32Array): boolean {
    const d = this.obsDim;
    const fl = this.frameLen;
    const cap = this.capacity;
    const oldest = (this.head - this.count + cap) % cap;
    if (this.count < L) return false;
    for (let b = 0; b < B; b++) {
      let k0 = -1;
      for (let tries = 0; tries < 40; tries++) {
        const k = rng.int(this.count - L + 1);
        const p0 = (oldest + k) % cap;
        const e = this.ep[p0];
        let ok = this.fo[p0] > this.nextFrameId - this.poolCap + 2;
        for (let t = 1; ok && t < L; t++) if (this.ep[(p0 + t) % cap] !== e) ok = false;
        if (ok) {
          k0 = k;
          break;
        }
      }
      if (k0 < 0) return false;
      for (let t = 0; t < L; t++) {
        const p = (oldest + k0 + t) % cap;
        const i = t * B + b;
        this.readFrame(this.fo[p], frames, i * fl);
        prop.set(this.obs.subarray(p * d, p * d + d), i * d);
        act[i] = this.action[p];
        rew[i] = this.reward[p];
        done[i] = this.done[p];
      }
      const pl = (oldest + k0 + L - 1) % cap;
      const i = L * B + b;
      this.readFrame(this.fn[pl], frames, i * fl);
      prop.set(this.next.subarray(pl * d, pl * d + d), i * d);
    }
    return true;
  }
}
