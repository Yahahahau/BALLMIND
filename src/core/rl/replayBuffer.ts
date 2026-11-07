import { Rng } from '../math/rng';

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
}
