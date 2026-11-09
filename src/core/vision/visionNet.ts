import { Rng } from '../math/rng';

/**
 * Multimodal Q-network: a small CNN "visual encoder" turns the stacked RGB frames
 * into a feature vector; that vector is concatenated with the physics (proprioceptive)
 * sensors and fed through dense layers to one Q-value per action.
 *
 *   frames ──conv/ReLU×N──► flatten ──dense/ReLU──► visual features ┐
 *   physics sensors ──────────────────────────────────────────────── ├─► dense/ReLU… ─► Q
 *
 * Everything (forward, back-propagation, parameters) is hand-written, dependency-free
 * and lives in one flat Float32Array so the DQN's Adam / target-sync code is trivial.
 */
export interface ConvSpec {
  k: number;
  s: number;
  out: number;
}

export interface VisionNetSpec {
  width: number;
  height: number;
  /** Input channels = 3 (RGB) × stacked frames. */
  channels: number;
  /** Number of physics sensor values appended after the visual features. */
  proprio: number;
  actions: number;
  convs: ConvSpec[];
  /** Size of the visual feature vector. */
  feat: number;
  hidden: number[];
}

interface ConvL {
  inC: number;
  inH: number;
  inW: number;
  outC: number;
  outH: number;
  outW: number;
  k: number;
  s: number;
  w: number;
  b: number;
}
interface DenseL {
  inN: number;
  outN: number;
  w: number;
  b: number;
  relu: boolean;
}

interface Ctx {
  B: number;
  conv: Float32Array[]; // post-ReLU activations of every conv layer
  feat: Float32Array;
  cat: Float32Array;
  dense: Float32Array[]; // activations of the head layers (last = Q)
  dConv: Float32Array[];
  dFeat: Float32Array;
  dCat: Float32Array;
  dDense: Float32Array[];
  x: Float32Array | null;
}

export class VisionNet {
  readonly spec: VisionNetSpec;
  readonly params: Float32Array;
  readonly convs: ConvL[] = [];
  readonly featL: DenseL;
  readonly head: DenseL[] = [];
  readonly flatSize: number;
  private ctxs = new Map<number, Ctx>();
  private last: Ctx | null = null;

  constructor(spec: VisionNetSpec, params?: Float32Array) {
    this.spec = spec;
    let n = 0;
    let c = spec.channels;
    let h = spec.height;
    let w = spec.width;
    for (const cv of spec.convs) {
      const oh = Math.floor((h - cv.k) / cv.s) + 1;
      const ow = Math.floor((w - cv.k) / cv.s) + 1;
      if (oh < 1 || ow < 1) throw new Error('vision net: image too small for conv stack');
      const wOff = n;
      n += cv.out * c * cv.k * cv.k;
      const bOff = n;
      n += cv.out;
      this.convs.push({ inC: c, inH: h, inW: w, outC: cv.out, outH: oh, outW: ow, k: cv.k, s: cv.s, w: wOff, b: bOff });
      c = cv.out;
      h = oh;
      w = ow;
    }
    this.flatSize = c * h * w;
    const dense = (inN: number, outN: number, relu: boolean): DenseL => {
      const l = { inN, outN, w: n, b: n + inN * outN, relu };
      n += inN * outN + outN;
      return l;
    };
    this.featL = dense(this.flatSize, spec.feat, true);
    let prev = spec.feat + spec.proprio;
    for (const hs of spec.hidden) {
      this.head.push(dense(prev, hs, true));
      prev = hs;
    }
    this.head.push(dense(prev, spec.actions, false));
    if (params && params.length !== n) throw new Error(`param size mismatch: ${params.length} != ${n}`);
    this.params = params ?? new Float32Array(n);
  }

  static paramCount(spec: VisionNetSpec): number {
    return new VisionNet(spec).params.length;
  }

  /** Multiply-accumulates of one forward pass of one sample (for benchmarking / docs). */
  get macs(): number {
    let m = 0;
    for (const l of this.convs) m += l.outC * l.outH * l.outW * l.inC * l.k * l.k;
    m += this.featL.inN * this.featL.outN;
    for (const l of this.head) m += l.inN * l.outN;
    return m;
  }

  init(rng: Rng): void {
    const p = this.params;
    p.fill(0);
    for (const l of this.convs) {
      const std = Math.sqrt(2 / (l.inC * l.k * l.k));
      for (let i = 0; i < l.outC * l.inC * l.k * l.k; i++) p[l.w + i] = rng.gaussian() * std;
    }
    const all = [this.featL, ...this.head];
    all.forEach((l, i) => {
      const std = Math.sqrt(2 / l.inN) * (i === all.length - 1 ? 0.1 : 1);
      for (let j = 0; j < l.inN * l.outN; j++) p[l.w + j] = rng.gaussian() * std;
    });
  }

  copyFrom(o: VisionNet): void {
    this.params.set(o.params);
  }

  private ctx(B: number): Ctx {
    let c = this.ctxs.get(B);
    if (c) return c;
    const sp = this.spec;
    const f = (n: number) => new Float32Array(B * n);
    c = {
      B,
      conv: this.convs.map((l) => f(l.outC * l.outH * l.outW)),
      feat: f(sp.feat),
      cat: f(sp.feat + sp.proprio),
      dense: this.head.map((l) => f(l.outN)),
      dConv: this.convs.map((l) => f(l.outC * l.outH * l.outW)),
      dFeat: f(sp.feat),
      dCat: f(sp.feat + sp.proprio),
      dDense: this.head.map((l) => f(l.outN)),
      x: null,
    };
    this.ctxs.set(B, c);
    return c;
  }

  /**
   * Batched forward pass. `frames` is B × (channels·H·W) in [0,1]; `proprio` is B × P.
   * Activations are cached for `backward`.
   */
  forwardBatch(frames: Float32Array, proprio: Float32Array, B: number): Float32Array {
    const ctx = this.ctx(B);
    this.last = ctx;
    ctx.x = frames;
    const p = this.params;
    let input = frames;
    for (let li = 0; li < this.convs.length; li++) {
      const l = this.convs[li];
      const out = ctx.conv[li];
      const inSz = l.inC * l.inH * l.inW;
      const outSz = l.outC * l.outH * l.outW;
      const kk = l.k * l.k;
      for (let b = 0; b < B; b++) {
        const ib = b * inSz;
        const ob = b * outSz;
        for (let oc = 0; oc < l.outC; oc++) {
          const bias = p[l.b + oc];
          const wBase = l.w + oc * l.inC * kk;
          for (let oy = 0; oy < l.outH; oy++) {
            for (let ox = 0; ox < l.outW; ox++) {
              let s = bias;
              const y0 = oy * l.s;
              const x0 = ox * l.s;
              for (let ic = 0; ic < l.inC; ic++) {
                const iBase = ib + ic * l.inH * l.inW;
                const wb = wBase + ic * kk;
                for (let ky = 0; ky < l.k; ky++) {
                  const row = iBase + (y0 + ky) * l.inW + x0;
                  const wr = wb + ky * l.k;
                  for (let kx = 0; kx < l.k; kx++) s += input[row + kx] * p[wr + kx];
                }
              }
              out[ob + (oc * l.outH + oy) * l.outW + ox] = s > 0 ? s : 0;
            }
          }
        }
      }
      input = out;
    }
    this.denseForward(this.featL, input, ctx.feat, B);
    const F = this.spec.feat;
    const P = this.spec.proprio;
    const cw = F + P;
    for (let b = 0; b < B; b++) {
      ctx.cat.set(ctx.feat.subarray(b * F, b * F + F), b * cw);
      for (let i = 0; i < P; i++) ctx.cat[b * cw + F + i] = proprio[b * P + i];
    }
    let a = ctx.cat;
    for (let i = 0; i < this.head.length; i++) {
      this.denseForward(this.head[i], a, ctx.dense[i], B);
      a = ctx.dense[i];
    }
    return a;
  }

  /** Single-sample inference (does not disturb the training caches). */
  predict(frames: Float32Array, proprio: Float32Array, out: Float32Array): void {
    const saved = this.last;
    const q = this.forwardBatch(frames, proprio, 1);
    out.set(q.subarray(0, this.spec.actions));
    this.last = saved;
  }

  /** The visual feature vector of the most recent forward pass (for inspection / tests). */
  lastFeatures(): Float32Array | null {
    return this.last ? this.last.feat : null;
  }

  private denseForward(l: DenseL, a: Float32Array, z: Float32Array, B: number): void {
    const p = this.params;
    for (let b = 0; b < B; b++) {
      const zb = b * l.outN;
      const ab = b * l.inN;
      for (let j = 0; j < l.outN; j++) z[zb + j] = p[l.b + j];
      for (let i = 0; i < l.inN; i++) {
        const ai = a[ab + i];
        if (ai === 0) continue;
        const row = l.w + i * l.outN;
        for (let j = 0; j < l.outN; j++) z[zb + j] += ai * p[row + j];
      }
      if (l.relu) for (let j = 0; j < l.outN; j++) if (z[zb + j] < 0) z[zb + j] = 0;
    }
  }

  /** dOut = dLoss/dQ (B × actions). Gradients are written into `grad` (overwritten). */
  backward(dOut: Float32Array, B: number, grad: Float32Array): void {
    const ctx = this.last;
    if (!ctx || ctx.B !== B || !ctx.x) throw new Error('vision net: backward without matching forward');
    const p = this.params;
    grad.fill(0);
    const nh = this.head.length;
    ctx.dDense[nh - 1].set(dOut.subarray(0, B * this.spec.actions));
    for (let i = nh - 1; i >= 0; i--) {
      const l = this.head[i];
      const aIn = i === 0 ? ctx.cat : ctx.dense[i - 1];
      const dIn = i === 0 ? ctx.dCat : ctx.dDense[i - 1];
      this.denseBackward(l, aIn, ctx.dDense[i], dIn, B, grad, i > 0 || true);
      if (i > 0) {
        // ReLU mask of the layer below
        const act = ctx.dense[i - 1];
        for (let j = 0; j < act.length; j++) if (act[j] <= 0) dIn[j] = 0;
      }
    }
    // split the concat gradient: visual part flows on, physics part stops here
    const F = this.spec.feat;
    const cw = F + this.spec.proprio;
    for (let b = 0; b < B; b++) ctx.dFeat.set(ctx.dCat.subarray(b * cw, b * cw + F), b * F);
    for (let j = 0; j < ctx.feat.length; j++) if (ctx.feat[j] <= 0) ctx.dFeat[j] = 0;

    const lastConv = this.convs.length - 1;
    this.denseBackward(this.featL, ctx.conv[lastConv], ctx.dFeat, ctx.dConv[lastConv], B, grad, true);
    for (let li = lastConv; li >= 0; li--) {
      const l = this.convs[li];
      const act = ctx.conv[li];
      const d = ctx.dConv[li];
      for (let j = 0; j < act.length; j++) if (act[j] <= 0) d[j] = 0;
      const input = li === 0 ? ctx.x : ctx.conv[li - 1];
      const dIn = li > 0 ? ctx.dConv[li - 1] : null;
      if (dIn) dIn.fill(0);
      const inSz = l.inC * l.inH * l.inW;
      const outSz = l.outC * l.outH * l.outW;
      const kk = l.k * l.k;
      for (let b = 0; b < B; b++) {
        const ib = b * inSz;
        const ob = b * outSz;
        for (let oc = 0; oc < l.outC; oc++) {
          const wBase = l.w + oc * l.inC * kk;
          for (let oy = 0; oy < l.outH; oy++) {
            for (let ox = 0; ox < l.outW; ox++) {
              const g = d[ob + (oc * l.outH + oy) * l.outW + ox];
              if (g === 0) continue;
              grad[l.b + oc] += g;
              const y0 = oy * l.s;
              const x0 = ox * l.s;
              for (let ic = 0; ic < l.inC; ic++) {
                const iBase = ib + ic * l.inH * l.inW;
                const wb = wBase + ic * kk;
                for (let ky = 0; ky < l.k; ky++) {
                  const row = iBase + (y0 + ky) * l.inW + x0;
                  const wr = wb + ky * l.k;
                  for (let kx = 0; kx < l.k; kx++) {
                    grad[wr + kx] += g * input![row + kx];
                    if (dIn) dIn[row + kx] += g * p[wr + kx];
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  /** Back-propagate through one dense layer; writes dIn (pre-mask) and accumulates grads. */
  private denseBackward(l: DenseL, a: Float32Array, d: Float32Array, dIn: Float32Array, B: number, grad: Float32Array, _needIn: boolean): void {
    const p = this.params;
    for (let b = 0; b < B; b++) {
      const db = b * l.outN;
      const ab = b * l.inN;
      for (let j = 0; j < l.outN; j++) grad[l.b + j] += d[db + j];
      for (let i = 0; i < l.inN; i++) {
        const ai = a[ab + i];
        const row = l.w + i * l.outN;
        let s = 0;
        for (let j = 0; j < l.outN; j++) {
          const dj = d[db + j];
          if (ai !== 0) grad[row + j] += ai * dj;
          s += dj * p[row + j];
        }
        dIn[ab + i] = s;
      }
    }
  }
}
