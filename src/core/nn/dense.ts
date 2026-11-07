import { Rng } from '../math/rng';

/**
 * Fully-connected network (ReLU hidden layers, linear output) with a flat
 * Float32Array parameter vector, batched forward pass and hand-written
 * back-propagation. No dependencies – runs inside a Web Worker.
 *
 * Layout per layer l: weights W[in*out] (row = input unit), then bias b[out].
 */
export class DenseNetwork {
  readonly sizes: number[];
  readonly params: Float32Array;
  private readonly wOff: number[] = [];
  private readonly bOff: number[] = [];
  private readonly single: Float32Array[];
  private acts: Float32Array[] = [];
  private deltaA: Float32Array = new Float32Array(0);
  private deltaB: Float32Array = new Float32Array(0);
  private batch = 0;

  constructor(sizes: number[], params?: Float32Array) {
    this.sizes = sizes.slice();
    let n = 0;
    for (let l = 0; l < sizes.length - 1; l++) {
      this.wOff.push(n);
      n += sizes[l] * sizes[l + 1];
      this.bOff.push(n);
      n += sizes[l + 1];
    }
    if (params && params.length !== n) throw new Error(`param size mismatch: ${params.length} != ${n}`);
    this.params = params ?? new Float32Array(n);
    this.single = sizes.map((s) => new Float32Array(s));
  }

  static paramCount(sizes: number[]): number {
    let n = 0;
    for (let l = 0; l < sizes.length - 1; l++) n += sizes[l] * sizes[l + 1] + sizes[l + 1];
    return n;
  }

  get layerCount(): number {
    return this.sizes.length - 1;
  }

  /** He initialisation; output layer is scaled down so initial Q-values are near 0. */
  init(rng: Rng): void {
    const L = this.layerCount;
    this.params.fill(0);
    for (let l = 0; l < L; l++) {
      const inN = this.sizes[l];
      const outN = this.sizes[l + 1];
      const std = Math.sqrt(2 / inN) * (l === L - 1 ? 0.1 : 1);
      const w = this.wOff[l];
      for (let i = 0; i < inN * outN; i++) this.params[w + i] = rng.gaussian() * std;
    }
  }

  copyFrom(other: DenseNetwork): void {
    this.params.set(other.params);
  }

  /** Single-sample inference. `out` receives the Q-values. */
  predict(x: Float32Array, out: Float32Array): void {
    const L = this.layerCount;
    this.single[0].set(x.subarray(0, this.sizes[0]));
    for (let l = 0; l < L; l++) {
      const inN = this.sizes[l];
      const outN = this.sizes[l + 1];
      const a = this.single[l];
      const z = this.single[l + 1];
      const w = this.wOff[l];
      const bo = this.bOff[l];
      for (let j = 0; j < outN; j++) z[j] = this.params[bo + j];
      for (let i = 0; i < inN; i++) {
        const ai = a[i];
        if (ai === 0) continue;
        const row = w + i * outN;
        for (let j = 0; j < outN; j++) z[j] += ai * this.params[row + j];
      }
      if (l < L - 1) for (let j = 0; j < outN; j++) if (z[j] < 0) z[j] = 0;
    }
    out.set(this.single[L].subarray(0, this.sizes[L]));
  }

  private ensureBatch(B: number): void {
    if (B === this.batch) return;
    this.batch = B;
    this.acts = this.sizes.map((s) => new Float32Array(B * s));
    const maxS = Math.max(...this.sizes);
    this.deltaA = new Float32Array(B * maxS);
    this.deltaB = new Float32Array(B * maxS);
  }

  /** Batched forward pass; activations are cached for `backward`. Returns B*out Q-values. */
  forwardBatch(x: Float32Array, B: number): Float32Array {
    this.ensureBatch(B);
    const L = this.layerCount;
    this.acts[0].set(x.subarray(0, B * this.sizes[0]));
    for (let l = 0; l < L; l++) {
      const inN = this.sizes[l];
      const outN = this.sizes[l + 1];
      const a = this.acts[l];
      const z = this.acts[l + 1];
      const w = this.wOff[l];
      const bo = this.bOff[l];
      const relu = l < L - 1;
      for (let b = 0; b < B; b++) {
        const zb = b * outN;
        const ab = b * inN;
        for (let j = 0; j < outN; j++) z[zb + j] = this.params[bo + j];
        for (let i = 0; i < inN; i++) {
          const ai = a[ab + i];
          if (ai === 0) continue;
          const row = w + i * outN;
          for (let j = 0; j < outN; j++) z[zb + j] += ai * this.params[row + j];
        }
        if (relu) for (let j = 0; j < outN; j++) if (z[zb + j] < 0) z[zb + j] = 0;
      }
    }
    return this.acts[L];
  }

  /**
   * Back-propagate dLoss/dOutput (B*out) through the activations cached by the
   * most recent `forwardBatch`. Gradients are written (overwritten) into `grad`.
   */
  backward(dOut: Float32Array, B: number, grad: Float32Array): void {
    const L = this.layerCount;
    grad.fill(0);
    let delta = this.deltaA;
    let next = this.deltaB;
    delta.set(dOut.subarray(0, B * this.sizes[L]));
    for (let l = L - 1; l >= 0; l--) {
      const inN = this.sizes[l];
      const outN = this.sizes[l + 1];
      const a = this.acts[l];
      const w = this.wOff[l];
      const bo = this.bOff[l];
      for (let b = 0; b < B; b++) {
        const db = b * outN;
        const ab = b * inN;
        for (let j = 0; j < outN; j++) grad[bo + j] += delta[db + j];
        for (let i = 0; i < inN; i++) {
          const ai = a[ab + i];
          if (ai === 0) continue;
          const row = w + i * outN;
          for (let j = 0; j < outN; j++) grad[row + j] += ai * delta[db + j];
        }
      }
      if (l > 0) {
        for (let b = 0; b < B; b++) {
          const db = b * outN;
          const ab = b * inN;
          for (let i = 0; i < inN; i++) {
            if (a[ab + i] <= 0) {
              next[ab + i] = 0;
              continue;
            }
            const row = w + i * outN;
            let s = 0;
            for (let j = 0; j < outN; j++) s += delta[db + j] * this.params[row + j];
            next[ab + i] = s;
          }
        }
        const t = delta;
        delta = next;
        next = t;
      }
    }
  }
}
