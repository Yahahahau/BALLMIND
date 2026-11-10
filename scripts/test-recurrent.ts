import { RecurrentVisionNet } from '../src/core/vision/recurrentNet';
import { Rng } from '../src/core/math/rng';
const net = new RecurrentVisionNet({ width: 12, height: 12, channels: 3, proprio: 3, convs: [{ k: 3, s: 2, out: 4 }, { k: 3, s: 1, out: 4 }], feat: 8, gru: 6, hidden: 8, moveActions: 3, gazeActions: 2 });
net.init(new Rng(5));
const rng = new Rng(11);
const T = 4, B = 2, fl = 3 * 144;
const x = new Float32Array(T * B * fl).map(() => rng.next());
const pr = new Float32Array(T * B * 3).map(() => rng.gaussian());
const tgt = new Float32Array(T * B * net.A).map(() => rng.gaussian());
const loss = () => { const q = net.unroll(x, pr, T, B); let l = 0; for (let i = 0; i < q.length; i++) l += 0.5 * (q[i] - tgt[i]) ** 2; return l; };
const q = net.unroll(x, pr, T, B);
const dQ = new Float32Array(q.length).map((_, i) => q[i] - tgt[i]);
const grad = new Float32Array(net.params.length);
net.backward(dQ, T, B, 0, grad);
const names: [string, number, number][] = [['conv0', net.convs[0].w, net.convs[0].b + 4], ['feat', net.featL.w, net.featL.b + net.featL.outN], ['gru', net.convs[1].b + 4 + net.featL.inN * 8 + 8, net.hidL.w], ['hid', net.hidL.w, net.hidL.b + 8], ['out', net.outL.w, net.outL.b + net.outL.outN]];
let bad = 0;
for (const [n, a, b] of names) {
  const errs: number[] = [];
  for (let t = 0; t < 80; t++) {
    const i = a + Math.floor(rng.next() * (b - a));
    const o = net.params[i]; const e = 1e-3;
    net.params[i] = o + e; const lp = loss(); net.params[i] = o - e; const lm = loss(); net.params[i] = o;
    const num = (lp - lm) / (2 * e);
    if (Math.abs(num) + Math.abs(grad[i]) < 1e-3) continue;
    errs.push(Math.abs(num - grad[i]) / (Math.abs(num) + Math.abs(grad[i])));
  }
  errs.sort((p, q) => p - q);
  const med = errs[errs.length >> 1] ?? 0;
  console.log(n.padEnd(5), errs.length, 'median rel err', med.toExponential(1));
  if (med > 2e-3) bad++;
}
// the recurrent state really matters: same frame, different history -> different Q
const h = new Float32Array(net.H); const o1 = new Float32Array(net.A), o2 = new Float32Array(net.A);
const f1 = x.subarray(0, fl), f2 = x.subarray(fl, 2 * fl), p0 = pr.subarray(0, 3);
net.predictStep(f1, p0, h, o1); net.predictStep(f2, p0, h, o1);
const h2 = new Float32Array(net.H); net.predictStep(f2, p0, h2, o2);
const d = o1.reduce((s, v, i) => s + Math.abs(v - o2[i]), 0);
console.log('memory effect on Q (same frame, different history):', d.toFixed(4));
if (d < 1e-4) bad++;
console.log(bad ? 'FAIL' : 'recurrent net OK');
process.exit(bad ? 1 : 0);
