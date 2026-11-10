import { Rng } from '../math/rng';
import type { ConvSpec } from './visionNet';
import { convBackward, convForward, denseBackward1, denseForward1, type ConvL, type DenseL } from './layers';

/**
 * CNN visual encoder → [visual features ‖ physics sensors] → GRU (temporal memory) → dense → two Q branches
 * (movement actions, gaze actions). Forward/backward (incl. back-propagation through time) are hand-written.
 */
export interface RecurrentSpec {
  width: number;
  height: number;
  channels: number; // 3 (RGB)
  proprio: number;
  convs: ConvSpec[];
  feat: number;
  gru: number;
  hidden: number;
  moveActions: number;
  gazeActions: number;
}

interface Step {
  conv: Float32Array[];
  feat: Float32Array;
  x: Float32Array; // GRU input = feat ‖ proprio
  hPrev: Float32Array;
  z: Float32Array;
  r: Float32Array;
  n: Float32Array;
  h: Float32Array;
  hid: Float32Array;
  q: Float32Array;
  // gradient scratch
  dConv: Float32Array[];
  dFeat: Float32Array;
  dx: Float32Array;
  dh: Float32Array;
  dHid: Float32Array;
  frames: Float32Array | null;
  fOff: number;
}

const sig = (x: number) => 1 / (1 + Math.exp(-x));

export class RecurrentVisionNet {
  readonly params: Float32Array;
  readonly convs: ConvL[] = [];
  readonly featL: DenseL;
  readonly hidL: DenseL;
  readonly outL: DenseL;
  readonly flat: number;
  readonly I: number;
  readonly H: number;
  readonly A: number;
  private wx: number;
  private uh: number;
  private gb: number;
  private stepPool = new Map<number, Step[]>();
  private act: Step | null = null;

  constructor(readonly spec: RecurrentSpec, params?: Float32Array) {
    let n = 0;
    let c = spec.channels;
    let h = spec.height;
    let w = spec.width;
    for (const cv of spec.convs) {
      const oh = Math.floor((h - cv.k) / cv.s) + 1;
      const ow = Math.floor((w - cv.k) / cv.s) + 1;
      if (oh < 1 || ow < 1) throw new Error('image too small for the conv stack');
      const wOff = n;
      n += cv.out * c * cv.k * cv.k;
      const bOff = n;
      n += cv.out;
      this.convs.push({ inC: c, inH: h, inW: w, outC: cv.out, outH: oh, outW: ow, k: cv.k, s: cv.s, w: wOff, b: bOff });
      c = cv.out;
      h = oh;
      w = ow;
    }
    this.flat = c * h * w;
    const dense = (inN: number, outN: number, relu: boolean): DenseL => {
      const l = { inN, outN, w: n, b: n + inN * outN, relu };
      n += inN * outN + outN;
      return l;
    };
    this.featL = dense(this.flat, spec.feat, true);
    this.I = spec.feat + spec.proprio;
    this.H = spec.gru;
    this.wx = n;
    n += this.I * 3 * this.H;
    this.uh = n;
    n += this.H * 3 * this.H;
    this.gb = n;
    n += 3 * this.H;
    this.hidL = dense(this.H, spec.hidden, true);
    this.A = spec.moveActions + spec.gazeActions;
    this.outL = dense(spec.hidden, this.A, false);
    if (params && params.length !== n) throw new Error('param size mismatch');
    this.params = params ?? new Float32Array(n);
  }

  get macs(): number {
    let m = 0;
    for (const l of this.convs) m += l.outC * l.outH * l.outW * l.inC * l.k * l.k;
    m += this.featL.inN * this.featL.outN + this.I * 3 * this.H + this.H * 3 * this.H + this.hidL.inN * this.hidL.outN + this.outL.inN * this.outL.outN;
    return m;
  }

  init(rng: Rng): void {
    const p = this.params;
    p.fill(0);
    for (const l of this.convs) {
      const std = Math.sqrt(2 / (l.inC * l.k * l.k));
      for (let i = 0; i < l.outC * l.inC * l.k * l.k; i++) p[l.w + i] = rng.gaussian() * std;
    }
    for (const [l, g] of [[this.featL, 1], [this.hidL, 1], [this.outL, 0.1]] as [DenseL, number][]) {
      const std = Math.sqrt(2 / l.inN) * g;
      for (let j = 0; j < l.inN * l.outN; j++) p[l.w + j] = rng.gaussian() * std;
    }
    const sx = Math.sqrt(1 / this.I);
    for (let i = 0; i < this.I * 3 * this.H; i++) p[this.wx + i] = rng.gaussian() * sx;
    const sh = Math.sqrt(1 / this.H);
    for (let i = 0; i < this.H * 3 * this.H; i++) p[this.uh + i] = rng.gaussian() * sh;
  }

  copyFrom(o: RecurrentVisionNet): void {
    this.params.set(o.params);
  }

  private makeStep(): Step {
    const f = (n: number) => new Float32Array(n);
    const sp = this.spec;
    return {
      conv: this.convs.map((l) => f(l.outC * l.outH * l.outW)),
      feat: f(sp.feat),
      x: f(this.I),
      hPrev: f(this.H),
      z: f(this.H),
      r: f(this.H),
      n: f(this.H),
      h: f(this.H),
      hid: f(sp.hidden),
      q: f(this.A),
      dConv: this.convs.map((l) => f(l.outC * l.outH * l.outW)),
      dFeat: f(sp.feat),
      dx: f(this.I),
      dh: f(this.H),
      dHid: f(sp.hidden),
      frames: null,
      fOff: 0,
    };
  }

  /** One step for one sample. `hPrev` null = zero state. Fills s.* (including s.q). */
  private stepForward(s: Step, frames: Float32Array, fOff: number, proprio: Float32Array, pOff: number, hPrev: Float32Array | null, hOff: number): void {
    const p = this.params;
    s.frames = frames;
    s.fOff = fOff;
    let input = frames;
    let inOff = fOff;
    for (let li = 0; li < this.convs.length; li++) {
      convForward(this.convs[li], p, input, inOff, s.conv[li], 0);
      input = s.conv[li];
      inOff = 0;
    }
    denseForward1(this.featL, p, input, 0, s.feat, 0);
    const F = this.spec.feat;
    s.x.set(s.feat, 0);
    for (let i = 0; i < this.spec.proprio; i++) s.x[F + i] = proprio[pOff + i];
    const H = this.H;
    if (hPrev) for (let j = 0; j < H; j++) s.hPrev[j] = hPrev[hOff + j];
    else s.hPrev.fill(0);
    // gates
    const I = this.I;
    const a = new Float32Array(3 * H); // small; per step
    for (let j = 0; j < 3 * H; j++) a[j] = p[this.gb + j];
    for (let i = 0; i < I; i++) {
      const xi = s.x[i];
      if (xi === 0) continue;
      const row = this.wx + i * 3 * H;
      for (let j = 0; j < 3 * H; j++) a[j] += xi * p[row + j];
    }
    for (let i = 0; i < H; i++) {
      const hi = s.hPrev[i];
      if (hi === 0) continue;
      const row = this.uh + i * 3 * H;
      for (let j = 0; j < 2 * H; j++) a[j] += hi * p[row + j];
    }
    for (let j = 0; j < H; j++) {
      s.z[j] = sig(a[j]);
      s.r[j] = sig(a[H + j]);
    }
    for (let i = 0; i < H; i++) {
      const rh = s.r[i] * s.hPrev[i];
      if (rh === 0) continue;
      const row = this.uh + i * 3 * H + 2 * H;
      for (let j = 0; j < H; j++) a[2 * H + j] += rh * p[row + j];
    }
    for (let j = 0; j < H; j++) {
      const nn = Math.tanh(a[2 * H + j]);
      s.n[j] = nn;
      s.h[j] = (1 - s.z[j]) * nn + s.z[j] * s.hPrev[j];
    }
    denseForward1(this.hidL, p, s.h, 0, s.hid, 0);
    denseForward1(this.outL, p, s.hid, 0, s.q, 0);
  }

  /** Inference for one sample with persistent hidden state `h` (updated in place). q → out. */
  predictStep(frame: Float32Array, proprio: Float32Array, h: Float32Array, out: Float32Array): void {
    const s = (this.act ??= this.makeStep());
    this.stepForward(s, frame, 0, proprio, 0, h, 0);
    h.set(s.h);
    out.set(s.q);
  }

  /**
   * Unroll T steps over B sequences (time-major: index t*B+b). Zero initial state.
   * Returns Q of every step ([T*B*A]) – valid until the next unroll with the same (T,B).
   */
  unroll(frames: Float32Array, proprio: Float32Array, T: number, B: number): Float32Array {
    const key = T * 1000 + B;
    let steps = this.stepPool.get(key);
    if (!steps) {
      steps = Array.from({ length: T * B }, () => this.makeStep());
      this.stepPool.set(key, steps);
    }
    const fl = this.spec.channels * this.spec.width * this.spec.height;
    const P = this.spec.proprio;
    const q = new Float32Array(T * B * this.A);
    for (let b = 0; b < B; b++) {
      for (let t = 0; t < T; t++) {
        const s = steps[t * B + b];
        const prev = t > 0 ? steps[(t - 1) * B + b] : null;
        this.stepForward(s, frames, (t * B + b) * fl, proprio, (t * B + b) * P, prev ? prev.h : null, 0);
        q.set(s.q, (t * B + b) * this.A);
      }
    }
    this.lastKey = key;
    return q;
  }
  private lastKey = 0;

  /**
   * Back-propagation through time. dQ is [T*B*A] (zeros where no loss). Steps t < fromT are treated as
   * burn-in (state warm-up only, no gradient). Gradients are written (overwritten) into `grad`.
   */
  backward(dQ: Float32Array, T: number, B: number, fromT: number, grad: Float32Array, toT = T): void {
    const steps = this.stepPool.get(T * 1000 + B);
    if (!steps || this.lastKey !== T * 1000 + B) throw new Error('backward without matching unroll');
    const p = this.params;
    grad.fill(0);
    const H = this.H;
    const I = this.I;
    const F = this.spec.feat;
    const da = new Float32Array(3 * H);
    for (let b = 0; b < B; b++) {
      let dhNext: Float32Array | null = null;
      for (let t = toT - 1; t >= fromT; t--) {
        const s = steps[t * B + b];
        const prev = t > 0 ? steps[(t - 1) * B + b] : null;
        // heads
        denseBackward1(this.outL, p, s.hid, 0, dQ, (t * B + b) * this.A, s.dHid, 0, grad);
        for (let j = 0; j < s.hid.length; j++) if (s.hid[j] <= 0) s.dHid[j] = 0;
        denseBackward1(this.hidL, p, s.h, 0, s.dHid, 0, s.dh, 0, grad);
        if (dhNext) for (let j = 0; j < H; j++) s.dh[j] += dhNext[j];
        // GRU
        const dhPrev = new Float32Array(H);
        const drh = new Float32Array(H);
        da.fill(0);
        for (let j = 0; j < H; j++) {
          const dh = s.dh[j];
          const z = s.z[j];
          const nn = s.n[j];
          const dn = dh * (1 - z);
          const dz = dh * (s.hPrev[j] - nn);
          dhPrev[j] = dh * z;
          da[2 * H + j] = dn * (1 - nn * nn);
          da[j] = dz * z * (1 - z);
        }
        // d(r∘h) = dan · Unᵀ
        for (let i = 0; i < H; i++) {
          const row = this.uh + i * 3 * H + 2 * H;
          let sum = 0;
          for (let j = 0; j < H; j++) sum += da[2 * H + j] * p[row + j];
          drh[i] = sum;
        }
        for (let i = 0; i < H; i++) {
          const r = s.r[i];
          const dr = drh[i] * s.hPrev[i];
          dhPrev[i] += drh[i] * r;
          da[H + i] = dr * r * (1 - r);
        }
        // parameter grads
        for (let j = 0; j < 3 * H; j++) grad[this.gb + j] += da[j];
        for (let i = 0; i < I; i++) {
          const xi = s.x[i];
          const row = this.wx + i * 3 * H;
          let dx = 0;
          for (let j = 0; j < 3 * H; j++) {
            if (xi !== 0) grad[row + j] += xi * da[j];
            dx += da[j] * p[row + j];
          }
          s.dx[i] = dx;
        }
        for (let i = 0; i < H; i++) {
          const hi = s.hPrev[i];
          const rh = s.r[i] * hi;
          const row = this.uh + i * 3 * H;
          let dhp = 0;
          for (let j = 0; j < 2 * H; j++) {
            if (hi !== 0) grad[row + j] += hi * da[j];
            dhp += da[j] * p[row + j];
          }
          if (rh !== 0) for (let j = 0; j < H; j++) grad[row + 2 * H + j] += rh * da[2 * H + j];
          dhPrev[i] += dhp;
        }
        dhNext = dhPrev;
        void prev;
        // encoder
        for (let j = 0; j < F; j++) s.dFeat[j] = s.feat[j] > 0 ? s.dx[j] : 0;
        const last = this.convs.length - 1;
        denseBackward1(this.featL, p, s.conv[last], 0, s.dFeat, 0, s.dConv[last], 0, grad);
        for (let li = last; li >= 0; li--) {
          const l = this.convs[li];
          const act = s.conv[li];
          const d = s.dConv[li];
          for (let j = 0; j < act.length; j++) if (act[j] <= 0) d[j] = 0;
          const dIn = li > 0 ? s.dConv[li - 1] : null;
          if (dIn) dIn.fill(0);
          const inp = li === 0 ? s.frames! : s.conv[li - 1];
          convBackward(l, p, inp, li === 0 ? s.fOff : 0, d, 0, dIn, grad);
        }
      }
    }
  }
}
