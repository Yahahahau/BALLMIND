import { VisionNet, VisionNetSpec } from '../src/core/vision/visionNet';
import { Rng } from '../src/core/math/rng';

// 1) numerical gradient check (float64 reference via small perturbations on a float32 net -> use loose tol)
const spec: VisionNetSpec = { width: 12, height: 12, channels: 6, proprio: 3, actions: 4, convs: [{ k: 3, s: 2, out: 4 }, { k: 3, s: 1, out: 4 }], feat: 8, hidden: [8] };
const net = new VisionNet(spec);
net.init(new Rng(3));
const rng = new Rng(9);
const B = 3;
const x = new Float32Array(B * 6 * 144).map(() => rng.next());
const pr = new Float32Array(B * 3).map(() => rng.gaussian());
const tgt = new Float32Array(B * 4).map(() => rng.gaussian());
const loss = () => { const q = net.forwardBatch(x, pr, B); let l = 0; for (let i = 0; i < q.length; i++) l += 0.5 * (q[i] - tgt[i]) ** 2; return l; };
const q = net.forwardBatch(x, pr, B);
const dOut = new Float32Array(q.length).map((_, i) => q[i] - tgt[i]);
const grad = new Float32Array(net.params.length);
net.backward(dOut, B, grad);
const errs: number[] = [];
const eps = 1e-3;
for (let t = 0; t < 600; t++) {
  const i = Math.floor(rng.next() * net.params.length);
  const o = net.params[i];
  net.params[i] = o + eps; const lp = loss();
  net.params[i] = o - eps; const lm = loss();
  net.params[i] = o;
  const num = (lp - lm) / (2 * eps);
  if (Math.abs(num) + Math.abs(grad[i]) < 1e-3) continue;
  errs.push(Math.abs(num - grad[i]) / (Math.abs(num) + Math.abs(grad[i])));
}
errs.sort((a, b) => a - b);
const med = errs[errs.length >> 1], p90 = errs[Math.floor(errs.length * 0.9)];
console.log(`gradient check: ${errs.length} params, median rel err ${med.toExponential(1)}, p90 ${p90.toExponential(1)} (outliers = ReLU kinks)`);
if (med > 1e-3 || p90 > 0.05) { console.error('FAIL gradient'); process.exit(1); }

// 2) benchmark candidate architectures
for (const [w, convs, feat] of [
  [32, [{ k: 4, s: 2, out: 8 }, { k: 3, s: 2, out: 16 }, { k: 3, s: 1, out: 16 }], 64],
  [40, [{ k: 5, s: 2, out: 8 }, { k: 3, s: 2, out: 16 }, { k: 3, s: 1, out: 16 }], 64],
  [48, [{ k: 5, s: 2, out: 8 }, { k: 3, s: 2, out: 16 }, { k: 3, s: 1, out: 16 }], 128],
] as const) {
  const s: VisionNetSpec = { width: w, height: w, channels: 6, proprio: 7, actions: 6, convs: convs as any, feat, hidden: [64] };
  const n = new VisionNet(s); n.init(new Rng(1));
  const Bn = 32;
  const xx = new Float32Array(Bn * 6 * w * w).map(() => Math.random());
  const pp = new Float32Array(Bn * 7);
  const d = new Float32Array(Bn * 6).fill(0.1);
  const g = new Float32Array(n.params.length);
  const t0 = performance.now();
  const R = 5;
  for (let r = 0; r < R; r++) { n.forwardBatch(xx, pp, Bn); n.forwardBatch(xx, pp, Bn); n.forwardBatch(xx, pp, Bn); n.backward(d, Bn, g); }
  const ms = (performance.now() - t0) / R;
  console.log(`${w}x${w}: params ${n.params.length}, MAC/fwd ${(n.macs / 1e3).toFixed(0)}k, ~${ms.toFixed(0)} ms per B=32 update (3 fwd + bwd, node)`);
}
