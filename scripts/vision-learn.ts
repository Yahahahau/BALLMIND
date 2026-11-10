// Headless learning attempt for the vision task using the CPU reference camera.
import { TrainingManager } from '../src/core/training/trainingManager';
import { getTask } from '../src/core/tasks/registry';
import { SoftReachRig } from '../src/core/vision/softRig';
import { DEFAULT_VISION, sanitizeVision } from '../src/core/vision/config';

const steps = Number(process.argv[2] ?? 20000);
const size = Number(process.argv[3] ?? 40);
const settings = sanitizeVision({ ...(getTask(process.argv[4] ?? 'reach-vision').vision?.defaults ?? {}), size });
const taskId = process.argv[4] ?? 'reach-vision';
const task = getTask(taskId);
const tm = await TrainingManager.create(task, 1, {}, { rig: new SoftReachRig(settings), settings });
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { modelFromJson, modelToJson } from '../src/core/training/model';
const ckpt = process.argv[5];
if (ckpt && existsSync(ckpt)) { tm.importModel(modelFromJson(readFileSync(ckpt, 'utf8'))); console.log('resumed from checkpoint at', tm.stats().envSteps, 'steps'); }
const budgetMs = Number(process.argv[6] ?? 0);
tm.start();
const t0 = Date.now();
let last = 0;
while (tm.stats().envSteps < steps) {
  tm.runDecisions(500);
  if (budgetMs && Date.now() - t0 > budgetMs) { writeFileSync(ckpt!, modelToJson(tm.exportModel(false))); console.log('checkpoint saved at', tm.stats().envSteps); process.exit(0); }
  const s = tm.stats();
  if (s.envSteps - last >= 2000) {
    last = s.envSteps;
    console.log(`steps ${s.envSteps} ep ${s.episodesDone} succ(last50) ${(s.successRate * 100).toFixed(0)}% avgR ${s.avgReward.toFixed(1)} eps ${s.epsilon.toFixed(2)} loss ${s.loss.toFixed(3)} train ${s.trainSteps} ${((Date.now() - t0) / 1000).toFixed(0)}s render ${s.vision?.renderMs.toFixed(2)}ms`);
  }
}
tm.setDemo(true);
tm.runDecisions(3000);
const s = tm.stats();
if (tm.gaze) console.log(`final gaze yaw ${tm.gaze.yaw.toFixed(0)}° pitch ${tm.gaze.pitch.toFixed(0)}°`);
console.log(`GREEDY (no exploration): ${s.demoEpisodes} episodes, success ${(s.demoSuccessRate * 100).toFixed(0)}%`);
