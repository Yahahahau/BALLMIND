import { Rng } from '../math/rng';

/** Compact (int8-quantised) snapshot of the replay memory for IndexedDB. Observations are clamped to ±3. */
export interface ReplayDump {
  n: number;
  obsDim: number;
  obs: Int8Array;
  next: Int8Array;
  action: Uint8Array;
  reward: Float32Array;
  done: Uint8Array;
}

const Q = 40;

/** Fixed-capacity ring buffer of transitions stored in typed arrays (no GC churn). */
export class ReplayBuffer {
  readonly capacity: number;
  readonly obsDim: number;
  private readonly obs: Float32Array;
  private readonly next: Float32Array;
  private readonly action: Int32Array;
  private readonly reward: Float32Array;
  private readonly done: Float32Array;
  private head = 0;
  private count = 0;

  constructor(capacity: number, obsDim: number) {
    this.capacity = capacity;
    this.obsDim = obsDim;
    this.obs = new Float32Array(capacity * obsDim);
    this.next = new Float32Array(capacity * obsDim);
    this.action = new Int32Array(capacity);
    this.reward = new Float32Array(capacity);
    this.done = new Float32Array(capacity);
  }

  get size(): number {
    return this.count;
  }

  clear(): void {
    this.head = 0;
    this.count = 0;
  }

  /** `terminal` = true only for real terminal states (not time-limit truncation). */
  add(obs: Float32Array, action: number, reward: number, next: Float32Array, terminal: boolean): void {
    const d = this.obsDim;
    const i = this.head;
    this.obs.set(obs.subarray(0, d), i * d);
    this.next.set(next.subarray(0, d), i * d);
    this.action[i] = action;
    this.reward[i] = reward;
    this.done[i] = terminal ? 1 : 0;
    this.head = (i + 1) % this.capacity;
    if (this.count < this.capacity) this.count++;
  }

  sample(
    B: number,
    rng: Rng,
    outObs: Float32Array,
    outAction: Int32Array,
    outReward: Float32Array,
    outNext: Float32Array,
    outDone: Float32Array,
  ): void {
    const d = this.obsDim;
    for (let b = 0; b < B; b++) {
      const i = rng.int(this.count);
      outObs.set(this.obs.subarray(i * d, i * d + d), b * d);
      outNext.set(this.next.subarray(i * d, i * d + d), b * d);
      outAction[b] = this.action[i];
      outReward[b] = this.reward[i];
      outDone[b] = this.done[i];
    }
  }

  /** Oldest -> newest, observations quantised to 1/40. */
  dump(): ReplayDump {
    const n = this.count;
    const d = this.obsDim;
    const out: ReplayDump = {
      n,
      obsDim: d,
      obs: new Int8Array(n * d),
      next: new Int8Array(n * d),
      action: new Uint8Array(n),
      reward: new Float32Array(n),
      done: new Uint8Array(n),
    };
    const start = this.count < this.capacity ? 0 : this.head;
    for (let k = 0; k < n; k++) {
      const i = (start + k) % this.capacity;
      for (let j = 0; j < d; j++) {
        out.obs[k * d + j] = Math.max(-127, Math.min(127, Math.round(this.obs[i * d + j] * Q)));
        out.next[k * d + j] = Math.max(-127, Math.min(127, Math.round(this.next[i * d + j] * Q)));
      }
      out.action[k] = this.action[i];
      out.reward[k] = this.reward[i];
      out.done[k] = this.done[i];
    }
    return out;
  }

  load(dump: ReplayDump): void {
    this.clear();
    const d = this.obsDim;
    if (dump.obsDim !== d) return;
    const tmpO = new Float32Array(d);
    const tmpN = new Float32Array(d);
    const from = Math.max(0, dump.n - this.capacity);
    for (let k = from; k < dump.n; k++) {
      for (let j = 0; j < d; j++) {
        tmpO[j] = dump.obs[k * d + j] / Q;
        tmpN[j] = dump.next[k * d + j] / Q;
      }
      this.add(tmpO, dump.action[k], dump.reward[k], tmpN, dump.done[k] === 1);
    }
  }
}
