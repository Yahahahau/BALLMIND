/**
 * Headless Obby training: real Rapier + real DQN + curriculum (no rendering).
 * Usage: npx tsx scripts/train-obby.ts [decisions=200000] [seed=1] [dqnOverridesJSON] [curriculumJSON]
 */
import { getTask } from '../src/core/tasks/registry';
import { TrainingManager } from '../src/core/training/trainingManager';

const total = Number(process.argv[2] ?? 200000);
const seed = Number(process.argv[3] ?? 1);
const dqn = process.argv[4] ? JSON.parse(process.argv[4]) : {};
const cur = process.argv[5] ? JSON.parse(process.argv[5]) : {};
const mgr = await TrainingManager.create(getTask('obby-challenge'), seed, dqn);
if (Object.keys(cur).length) mgr.command('curriculum', cur);
console.log('dqn', JSON.stringify(mgr.agent.config), 'curriculum', JSON.stringify(cur));

const t0 = performance.now();
const chunk = 10000;
for (let done = 0; done < total; done += chunk) {
  mgr.runDecisions(chunk);
  const s = mgr.stats();
  const st = (s.extra!.stages as any[]).map((x, i) => `S${i + 1}:${x.successes}/${x.episodes} p${(x.avgProgress * 100).toFixed(0)}`).join(' ');
  console.log(
    `steps=${s.envSteps} ep=${s.episodesDone} unl=${(s.extra!.unlocked as number) + 1} eps=${s.epsilon.toFixed(2)} loss=${s.loss.toFixed(3)} avgR=${s.avgReward.toFixed(1)} | ${st} (${((performance.now() - t0) / 1000).toFixed(0)}s)`,
  );
}
