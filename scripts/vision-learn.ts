// Headless learning attempt for the vision task using the CPU reference camera.
import { TrainingManager } from '../src/core/training/trainingManager';
import { getTask } from '../src/core/tasks/registry';
import { SoftReachRig } from '../src/core/vision/softRig';
import { DEFAULT_VISION, sanitizeVision } from '../src/core/vision/config';

const steps = Number(process.argv[2] ?? 20000);
const size = Number(process.argv[3] ?? 40);
const settings = sanitizeVision({ size });
const task = getTask('reach-vision');
const tm = await TrainingManager.create(task, 1, {}, { rig: new SoftReachRig(settings), settings });
tm.start();
const t0 = Date.now();
let last = 0;
while (tm.stats().envSteps < steps) {
  tm.runDecisions(500);
  const s = tm.stats();
  if (s.envSteps - last >= 2000) {
    last = s.envSteps;
    console.log(`steps ${s.envSteps} ep ${s.episodesDone} succ(last50) ${(s.successRate * 100).toFixed(0)}% avgR ${s.avgReward.toFixed(1)} eps ${s.epsilon.toFixed(2)} loss ${s.loss.toFixed(3)} train ${s.trainSteps} ${((Date.now() - t0) / 1000).toFixed(0)}s render ${s.vision?.renderMs.toFixed(2)}ms`);
  }
}
tm.setDemo(true);
tm.runDecisions(3000);
const s = tm.stats();
console.log(`GREEDY (no exploration): ${s.demoEpisodes} episodes, success ${(s.demoSuccessRate * 100).toFixed(0)}%`);
