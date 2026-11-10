import { SnapshotInterpolator } from '../src/render/snapshotInterpolator';
let fails = 0;
const ok = (c: boolean, m: string) => { console.log(`${c ? 'PASS' : 'FAIL'}  ${m}`); if (!c) fails++; };

// Physics: ball moves at 3 m/s along x in fixed 1/60 s steps. The worker ticks every ~16 ms but a tick may
// run 0, 1 or 2 steps (the exact jitter source of the old exponential-chase renderer), arrival times are noisy.
function run(smooth: 'interp' | 'chase') {
  const it = new SnapshotInterpolator();
  let wall = 0, simSteps = 0, acc = 0, seed = 7;
  const rnd = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);
  const arrivals: { at: number; t: number }[] = [];
  let nextTick = 0;
  for (let i = 0; i < 400; i++) { // 400 worker ticks ≈ 6.7 s
    nextTick += 16 + (rnd() < 0.15 ? 14 : 0) + rnd() * 3; // occasional slow tick (training update)
    acc += (nextTick - wall) / 1000; wall = nextTick;
    while (acc >= 1 / 60) { acc -= 1 / 60; simSteps++; }
    arrivals.push({ at: wall, t: simSteps / 60 });
  }
  // render at 60 fps (16.67 ms)
  let idx = 0, chasePos = 0, snapPos = 0;
  const shown: number[] = [];
  const truth: number[] = [];
  for (let f = 0; f * 16.667 < wall - 50; f++) {
    const now = f * 16.667;
    while (idx < arrivals.length && arrivals[idx].at <= now) {
      const a = arrivals[idx++];
      snapPos = 3 * a.t;
      it.push({ t: a.t, p: [snapPos, 0.5, 0], q: [0, 0, 0, 1], epoch: 0 }, a.at);
    }
    if (!it.ready) continue;
    let x: number;
    if (smooth === 'interp') x = it.sample(now)!.p[0];
    else { chasePos += (snapPos - chasePos) * (1 - Math.exp(-38 * 0.01667)); x = chasePos; }
    if (f > 40) { shown.push(x); }
  }
  // per-frame displacement variation (jitter) – ideal constant 3 m/s * 16.667 ms = 0.05 m
  const d = shown.slice(1).map((v, i) => v - shown[i]);
  const mean = d.reduce((a, b) => a + b, 0) / d.length;
  const sd = Math.sqrt(d.reduce((a, b) => a + (b - mean) ** 2, 0) / d.length);
  return { mean, sd, min: Math.min(...d), stats: it.stats };
}
const a = run('chase'), b = run('interp');
console.log('old exponential chase  : step mean', a.mean.toFixed(4), 'sd', a.sd.toFixed(4), 'min', a.min.toFixed(4));
console.log('snapshot interpolation: step mean', b.mean.toFixed(4), 'sd', b.sd.toFixed(4), 'min', b.min.toFixed(4), 'lag(s)', b.stats.lag.toFixed(3));
ok(b.sd < a.sd * 0.6, `frame-to-frame step jitter reduced (sd ${a.sd.toFixed(4)} → ${b.sd.toFixed(4)} m)`);
ok(b.min >= 0, 'interpolated motion never runs backwards');
ok(b.stats.lag < 0.12, 'render lag stays under 120 ms of physics time');

// teleport (epoch change) must snap, not slide
const it = new SnapshotInterpolator();
it.push({ t: 0, p: [0, 0, 0], q: [0, 0, 0, 1], epoch: 0 }, 0);
it.push({ t: 1 / 60, p: [0.05, 0, 0], q: [0, 0, 0, 1], epoch: 0 }, 16);
it.push({ t: 2 / 60, p: [50, 0, 0], q: [0, 0, 0, 1], epoch: 1 }, 33);
const p = it.sample(34)!;
ok(Math.abs(p.p[0] - 50) < 1e-6, 'epoch change snaps to the new position');
// yaw interpolation takes the short way round ±180
const it2 = new SnapshotInterpolator();
it2.push({ t: 0, p: [0, 0, 0], q: [0, 0, 0, 1], epoch: 0, yaw: 170, pitch: 0 }, 0);
it2.push({ t: 0.1, p: [0, 0, 0], q: [0, 0, 0, 1], epoch: 0, yaw: -170, pitch: 0 }, 100);
it2.push({ t: 0.2, p: [0, 0, 0], q: [0, 0, 0, 1], epoch: 0, yaw: -150, pitch: 0 }, 200);
const q2 = it2.sample(150)!;
ok(Math.abs(Math.abs(q2.yaw!) - 175) < 10 || Math.abs(q2.yaw!) > 150, `yaw interpolation wraps the short way (${q2.yaw!.toFixed(0)}°)`);
process.exit(fails ? 1 : 0);
