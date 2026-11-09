import { TrainingManager } from '../src/core/training/trainingManager';
import { getTask } from '../src/core/tasks/registry';
import { SoftReachRig } from '../src/core/vision/softRig';
import { sanitizeVision } from '../src/core/vision/config';
import { modelFromJson, modelToJson } from '../src/core/training/model';

let fails = 0;
const ok = (c: boolean, m: string) => { console.log(`${c ? 'PASS' : 'FAIL'}  ${m}`); if (!c) fails++; };

const settings = sanitizeVision({ size: 24, stack: 2 });
const task = getTask('reach-vision');
const mk = (seed: number, st = settings) => TrainingManager.create(task, seed, { minReplay: 200, batchSize: 16, bufferCapacity: 800, updatesPerStep: 1 }, { rig: new SoftReachRig(st), settings: st });

// 1. observation holds physics sensors only (no coordinates of target / ball)
const a = await mk(1);
const b = await mk(2); // different seed => different target position
ok(a.env.observationSize === 8, 'vision observation = 8 physics-sensor values (no coordinates)');
const sa = a.env.snapshot().bodies.target.p, sb = b.env.snapshot().bodies.target.p;
a.runDecisions(1); b.runDecisions(1);
ok(sa[0] !== sb[0] || sa[2] !== sb[2], 'the two episodes have different target positions');

// 2. the camera image really depends on the target position
const rig = new SoftReachRig(settings);
const f1 = new Uint8Array(3 * 24 * 24), f2 = new Uint8Array(3 * 24 * 24);
rig.render(a.env.snapshot(), f1);
rig.render(b.env.snapshot(), f2);
let d = 0; for (let i = 0; i < f1.length; i++) if (f1[i] !== f2[i]) d++;
ok(d > 5, `AI camera images differ with target position (${d} differing values)`);

// 3. train: replay fills, loss finite, weights change
const w0 = a.exportModel().online.slice(0, 50);
a.start(); a.runDecisions(1200);
const s = a.stats();
const w1 = a.exportModel().online.slice(0, 50);
ok(s.trainSteps > 100 && Number.isFinite(s.loss), `trained ${s.trainSteps} updates, loss ${s.loss.toFixed(4)}`);
ok(w0.some((v, i) => v !== w1[i]), 'CNN weights changed through learning');
ok(s.bufferSize === 800, 'replay ring capped at capacity');

// 4. save / load round trip incl. architecture check
const blob = a.exportModel(true);
ok(blob.replay === undefined, 'vision replay frames are not persisted');
const back = modelFromJson(modelToJson(blob));
const c = await mk(3);
c.importModel(back);
ok(c.exportModel().online[7] === a.exportModel().online[7], 'model import restores weights');
let rejected = false;
const other = sanitizeVision({ size: 32, stack: 2 });
const d2 = await mk(4, other);
try { d2.importModel(back); } catch { rejected = true; }
ok(rejected, 'model with different image size rejected');

// 5. demo mode (greedy) runs without learning
const before = a.stats().trainSteps;
a.setDemo(true); a.runDecisions(100);
ok(a.stats().trainSteps === before, 'demo mode does not train');
console.log(fails ? `${fails} FAILED` : 'all vision tests passed');
process.exit(fails ? 1 : 0);
