/** Physics / system tests for the Obby environment (no learning involved). */
import { ObbyEnvironment } from '../src/core/tasks/obby/environment';
import { initPhysics } from '../src/core/physics/physicsWorld';
import { STAGES } from '../src/core/tasks/obby/levels/compile';

await initPhysics();
let failed = 0;
const check = (name: string, ok: boolean, info = '') => {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name} ${info}`);
  if (!ok) failed++;
};
const idx = (env: ObbyEnvironment, id: string) => env.actionSet.actions.findIndex((a) => a.id === id);

function make(stage: number, seed = 1) {
  const env = new ObbyEnvironment(seed);
  env.curriculum.config = { ...env.curriculum.config, auto: false, manualStage: stage };
  env.reset();
  return env;
}
function run(env: ObbyEnvironment, action: string, decisions: number, startDecision = 0) {
  let last: ReturnType<ObbyEnvironment['evaluate']> | null = null;
  const a = idx(env, action);
  for (let d = 0; d < decisions; d++) {
    for (let s = 0; s < 6; s++) env.physicsStep(a, s === 0);
    last = env.evaluate(startDecision + d);
    if (last.done) break;
  }
  return last!;
}
const ballPos = (env: any) => env.ball.body.translation();

// 1. observation finite + size; ball rests on start platform
{
  const env = make(0);
  const r = run(env, 'neutral', 30);
  const p = ballPos(env);
  check('obs size', r.obs.length === env.observationSize, `(${r.obs.length})`);
  check('obs finite & bounded', Array.from(r.obs).every((v) => Number.isFinite(v) && Math.abs(v) <= 1.5 + 1e-6));
  check('ball rests on start platform', Math.abs(p.y - 0.5) < 0.05 && Math.hypot(p.x, p.z) < 0.2, `y=${p.y.toFixed(3)}`);
  check('grounded', (env as any).grounded === true);
}

// 2. stage geometry: every stage builds, ball spawns and rests
for (let s = 0; s < STAGES.length; s++) {
  const env = make(s);
  run(env, 'neutral', 20);
  const p = ballPos(env);
  check(`stage ${s + 1} builds + ball rests`, Math.abs(p.y - 0.5) < 0.06, `y=${p.y.toFixed(3)}`);
  env.dispose();
}

// 3. forward movement + jump uses physics (velocity changes via impulses only)
{
  const env = make(0);
  run(env, 'forward', 10);
  const v = (env as any).ball.body.linvel();
  check('forward accelerates (physics)', v.z < -3, `vz=${v.z.toFixed(2)}`);
}

// 4. fall -> penalty, respawn at start (no checkpoint yet), run continues, replay buffer data intact
{
  const env = make(0);
  let fellAt = -1;
  let rewardOnFall = 0;
  let r: any;
  for (let d = 0; d < 200; d++) {
    for (let s = 0; s < 6; s++) env.physicsStep(idx(env, 'backward'), s === 0);
    r = env.evaluate(d);
    if (r.failed) { fellAt = d; rewardOnFall = r.reward; break; }
  }
  const p = ballPos(env);
  check('falls off the back of the start platform', fellAt > 0, `decision ${fellAt}`);
  check('fall is terminal with penalty', r.terminal && rewardOnFall < -5, `reward=${rewardOnFall.toFixed(2)}`);
  check('respawned (episode continues)', r.respawned === true && r.done === false);
  check('respawn on start platform', Math.hypot(p.x, p.z) < 0.5 && p.y > 0.4 && p.y < 0.8, `(${p.x.toFixed(2)},${p.y.toFixed(2)},${p.z.toFixed(2)})`);
}

// 5. reward exploit check: hopping off and respawning repeatedly yields no positive net reward
{
  const env = make(0);
  let total = 0;
  let falls = 0;
  for (let d = 0; d < 600 && falls < 5; d++) {
    for (let s = 0; s < 6; s++) env.physicsStep(idx(env, 'forward_left'), s === 0);
    const r = env.evaluate(d);
    total += r.reward;
    if (r.failed) falls++;
    if (r.done) break;
  }
  check('suicide loop never pays', total < 0, `total=${total.toFixed(1)} after ${falls} falls`);
}

// 6. moving platform: drags the ball (friction) AND a simple feedback push can stay on it (controllable)
{
  const env = make(2);
  const e: any = env;
  const top = (i: number) => { const k = e.plats[i].body.translation(); return { x: k.x, z: k.z, y: k.y + STAGES[2].def.platforms[i].thick / 2 }; };
  const m = top(1);
  e.ball.teleport(m.x, m.y + 0.55, m.z);
  let maxDrag = 0;
  for (let d = 0; d < 6; d++) { for (let s = 0; s < 6; s++) env.physicsStep(idx(env, 'neutral'), s === 0); env.evaluate(d); const v = e.ball.body.linvel(); maxDrag = Math.max(maxDrag, Math.hypot(v.x, v.z)); }
  check('moving platform drags the ball (contact friction)', maxDrag > 0.3, `ball speed picked up ${maxDrag.toFixed(2)} m/s`);
  // controllability: test-only feedback controller (NOT part of the product) keeps the ball on the platform for 12 s
  const env2 = make(2); const e2: any = env2;
  const k0 = e2.plats[1].body.translation(); e2.ball.teleport(k0.x, k0.y + 0.9, k0.z);
  const dirs = env2.actionSet.actions.map((a) => a.move);
  let stayed = 0;
  for (let d = 0; d < 120; d++) {
    const pk = e2.plats[1].body.translation(); const b = e2.ball.body.translation(); const v = e2.ball.body.linvel();
    const wx = pk.x - b.x - 0.4 * v.x, wz = pk.z - b.z - 0.4 * v.z;
    let best = 0, bs = -9; env2.actionSet.actions.forEach((a, i) => { if (a.jump) return; const sc = a.move[0] * wx + a.move[1] * wz; if (sc > bs) { bs = sc; best = i; } });
    void dirs;
    for (let s = 0; s < 6; s++) env2.physicsStep(best, s === 0);
    const r = env2.evaluate(d); if (r.failed) break; stayed = d + 1;
  }
  check('ball can be controlled to stay on a moving platform', stayed === 120, `stayed ${(stayed / 10).toFixed(1)} s of 12`);
}

// 7. rotating platform drags the ball around its centre in the platform's sense of rotation
{
  const env = make(3);
  const e: any = env;
  const c = STAGES[3].def.platforms[1];
  const k = e.plats[1].body.translation();
  e.ball.teleport(k.x + 1.4, c.pos[1] + 0.55, k.z);
  run(env, 'neutral', 3);
  const p0 = ballPos(env);
  const a0 = Math.atan2(p0.z - k.z, p0.x - k.x);
  run(env, 'neutral', 30, 3);
  const p1 = ballPos(env);
  let da = Math.atan2(p1.z - k.z, p1.x - k.x) - a0; while (da > Math.PI) da -= 2 * Math.PI; while (da < -Math.PI) da += 2 * Math.PI;
  // yaw +ω about +Y turns atan2(z,x) negative
  check('rotating platform drags the ball in its rotation direction', da < -0.15 && p1.y > 0.4, `swept ${(da * 180 / Math.PI).toFixed(0)}° in 3 s (platform turns 103°; a free sphere lags by physics)`);
}

// 8. spinner bar pushes the ball (stage 4, hazard 0 on platform 3)
{
  const env = make(3);
  const e: any = env;
  const pl = STAGES[3].def.platforms[3];
  e.ball.teleport(pl.pos[0] + 1.4, pl.pos[1] + 0.55, pl.pos[2] - 0.0);
  let maxSpeed = 0;
  for (let d = 0; d < 60; d++) {
    for (let s = 0; s < 6; s++) env.physicsStep(idx(env, 'neutral'), s === 0);
    env.evaluate(d);
    const v = e.ball.body.linvel();
    maxSpeed = Math.max(maxSpeed, Math.hypot(v.x, v.z));
  }
  check('spinner bar collides with / pushes the ball', maxSpeed > 1.0, `peak speed ${maxSpeed.toFixed(2)} m/s`);
}

// 9. slippery surface has much less grip than normal
{
  const stopDist = (stage: number, platIdx: number) => {
    const env = make(stage);
    const e: any = env;
    const p = STAGES[stage].def.platforms[platIdx];
    const k = e.plats[platIdx].body.translation();
    const f = [-Math.sin(p.yaw), -Math.cos(p.yaw)];
    e.ball.teleport(p.pos[0] - f[0] * 1.5, p.pos[1] + 0.55, p.pos[2] - f[1] * 1.5);
    e.ball.body.setLinvel({ x: f[0] * 5, y: 0, z: f[1] * 5 }, true);
    run(env, 'neutral', 8);
    const v = e.ball.body.linvel();
    void k;
    return Math.hypot(v.x, v.z);
  };
  const ice = stopDist(3, 5);   // slippery
  const grip = stopDist(3, 2);  // normal
  check('slippery platform keeps momentum (low friction)', ice > grip + 0.3, `speed after 0.8 s: ice ${ice.toFixed(2)} vs normal ${grip.toFixed(2)} (note: rolling ball, friction affects spin transfer)`);
}

// 10. determinism of the fixed timestep
{
  const sim = () => {
    const env = make(0, 7);
    for (let d = 0; d < 40; d++) { for (let s = 0; s < 6; s++) env.physicsStep(1 + (d % 8), s === 0); env.evaluate(d); }
    const p = ballPos(env); return [p.x, p.y, p.z];
  };
  const a = sim(); const b = sim();
  check('deterministic physics', a.every((v, i) => Math.abs(v - b[i]) < 1e-9), JSON.stringify(a.map((v) => +v.toFixed(3))));
}

// 11. speed benchmark: random policy
{
  const env = make(0);
  const n = 3000; const t = performance.now();
  for (let d = 0; d < n; d++) { for (let s = 0; s < 6; s++) env.physicsStep(Math.floor(Math.random() * 15), s === 0); const r = env.evaluate(d % 600); if (r.done) env.reset(); }
  console.log(`INFO  physics+sensors speed: ${(n / ((performance.now() - t) / 1000)).toFixed(0)} decisions/s (no learning)`);
}

console.log(failed ? `\n${failed} test(s) FAILED` : '\nall physics tests passed');
process.exit(failed ? 1 : 0);
