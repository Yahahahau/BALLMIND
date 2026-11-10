/** Stand-alone conv / dense kernels shared by the recurrent vision network (flat Float32Array params). */
export interface ConvL {
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
export interface DenseL {
  inN: number;
  outN: number;
  w: number;
  b: number;
  relu: boolean;
}

export function convForward(l: ConvL, p: Float32Array, input: Float32Array, inOff: number, out: Float32Array, outOff: number): void {
  const kk = l.k * l.k;
  for (let oc = 0; oc < l.outC; oc++) {
    const bias = p[l.b + oc];
    const wBase = l.w + oc * l.inC * kk;
    for (let oy = 0; oy < l.outH; oy++) {
      for (let ox = 0; ox < l.outW; ox++) {
        let s = bias;
        const y0 = oy * l.s;
        const x0 = ox * l.s;
        for (let ic = 0; ic < l.inC; ic++) {
          const iBase = inOff + ic * l.inH * l.inW;
          const wb = wBase + ic * kk;
          for (let ky = 0; ky < l.k; ky++) {
            const row = iBase + (y0 + ky) * l.inW + x0;
            const wr = wb + ky * l.k;
            for (let kx = 0; kx < l.k; kx++) s += input[row + kx] * p[wr + kx];
          }
        }
        out[outOff + (oc * l.outH + oy) * l.outW + ox] = s > 0 ? s : 0;
      }
    }
  }
}

/** d must already be ReLU-masked. dIn (optional) receives += gradient wrt the layer input. */
export function convBackward(
  l: ConvL,
  p: Float32Array,
  input: Float32Array,
  inOff: number,
  d: Float32Array,
  dOff: number,
  dIn: Float32Array | null,
  grad: Float32Array,
): void {
  const kk = l.k * l.k;
  for (let oc = 0; oc < l.outC; oc++) {
    const wBase = l.w + oc * l.inC * kk;
    for (let oy = 0; oy < l.outH; oy++) {
      for (let ox = 0; ox < l.outW; ox++) {
        const g = d[dOff + (oc * l.outH + oy) * l.outW + ox];
        if (g === 0) continue;
        grad[l.b + oc] += g;
        const y0 = oy * l.s;
        const x0 = ox * l.s;
        for (let ic = 0; ic < l.inC; ic++) {
          const iBase = inOff + ic * l.inH * l.inW;
          const wb = wBase + ic * kk;
          for (let ky = 0; ky < l.k; ky++) {
            const row = iBase + (y0 + ky) * l.inW + x0;
            const wr = wb + ky * l.k;
            for (let kx = 0; kx < l.k; kx++) {
              grad[wr + kx] += g * input[row + kx];
              if (dIn) dIn[row + kx] += g * p[wr + kx];
            }
          }
        }
      }
    }
  }
}

/** z = a·W + b for one sample (offsets into flat arrays). Optional ReLU. */
export function denseForward1(l: DenseL, p: Float32Array, a: Float32Array, aOff: number, z: Float32Array, zOff: number): void {
  for (let j = 0; j < l.outN; j++) z[zOff + j] = p[l.b + j];
  for (let i = 0; i < l.inN; i++) {
    const ai = a[aOff + i];
    if (ai === 0) continue;
    const row = l.w + i * l.outN;
    for (let j = 0; j < l.outN; j++) z[zOff + j] += ai * p[row + j];
  }
  if (l.relu) for (let j = 0; j < l.outN; j++) if (z[zOff + j] < 0) z[zOff + j] = 0;
}

/** d (pre-masked) → grads, dIn = d·Wᵀ (overwritten). */
export function denseBackward1(
  l: DenseL,
  p: Float32Array,
  a: Float32Array,
  aOff: number,
  d: Float32Array,
  dOff: number,
  dIn: Float32Array | null,
  dInOff: number,
  grad: Float32Array,
): void {
  for (let j = 0; j < l.outN; j++) grad[l.b + j] += d[dOff + j];
  for (let i = 0; i < l.inN; i++) {
    const ai = a[aOff + i];
    const row = l.w + i * l.outN;
    let s = 0;
    for (let j = 0; j < l.outN; j++) {
      const dj = d[dOff + j];
      if (ai !== 0) grad[row + j] += ai * dj;
      s += dj * p[row + j];
    }
    if (dIn) dIn[dInOff + i] = s;
  }
}
