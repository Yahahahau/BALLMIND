// Evaluate a trained first-person checkpoint greedily: real camera vs. blinded camera (constant grey image).
import { readFileSync } from 'node:fs';
import { TrainingManager } from '../src/core/training/trainingManager';
import { getTask } from '../src/core/tasks/registry';
import { SoftReachRig } from '../src/core/vision/softRig';
import { sanitizeVision } from '../src/core/vision/config';
import { modelFromJson } from '../src/core/training/model';
import type { VisionRig } from '../src/core/vision/rig';

const task = getTask('reach-fp');
const settings = sanitizeVision({ ...task.vision!.defaults, size: 48 });
const ckpt = modelFromJson(readFileSync(process.argv[2], 'utf8'));
const episodes = Number(process.argv[3] ?? 60);
async function evalOnce(blind: boolean) {
  const inner = new SoftReachRig(settings);
  const rig: VisionRig = blind ? { kind: 'blind', get settings() { return inner.settings; }, set settings(v) { inner.settings = v; }, configure: (s) => inner.configure(s), dispose() {}, render(snap, out, pose) { inner.render(snap, out, pose); out.fill(128); } } : inner;
  const tm = await TrainingManager.create(task, 77, {}, { rig, settings });
  tm.importModel(ckpt);
  tm.setDemo(true); tm.start();
  let guard = 0;
  while (tm.stats().demoEpisodes < episodes && guard++ < 400) tm.runDecisions(100);
  const s = tm.stats();
  return `${blind ? 'BLINDED camera ' : 'real camera    '}: ${s.demoEpisodes} episodes, success ${(s.demoSuccessRate * 100).toFixed(0)}%`;
}
console.log(await evalOnce(false));
console.log(await evalOnce(true));
