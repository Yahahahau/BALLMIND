/**
 * Headless learning test: trains the real DQN against the real Rapier physics
 * (no rendering) and reports whether performance improves over a random policy.
 * Usage: npx tsx scripts/train-headless.ts [decisions=60000] [seed=1]
 */
import { getTask } from '../src/core/tasks/registry';
import { TrainingManager } from '../src/core/training/trainingManager';

const total = Number(process.argv[2] ?? 60000);
const seed = Number(process.argv[3] ?? 1);

const task = getTask('reach-square');
const overrides = process.argv[4] ? JSON.parse(process.argv[4]) : {};
const mgr = await TrainingManager.create(task, seed, overrides);

async function evaluate(label: string, episodes = 100): Promise<number> {
  mgr.setDemo(true);
  mgr.runDecisions(1);
  while (mgr.stats().demoEpisodes < episodes) mgr.runDecisions(50);
  const s = mgr.stats();
  console.log(`  [${label}] greedy success ${(s.demoSuccessRate * 100).toFixed(0)}% over ${s.demoEpisodes} episodes`);
  const rate = s.demoSuccessRate;
  mgr.setDemo(false);
  return rate;
}

console.log(JSON.stringify(overrides));
console.log(`task=${task.info.id} obs=${mgr.env.observationSize} actions=${mgr.env.actionSet.size} seed=${seed}`);
const before = await evaluate('untrained');

const t0 = performance.now();
const chunk = 2000;
for (let done = 0; done < total; done += chunk) {
  mgr.runDecisions(chunk);
  const s = mgr.stats();
  console.log(
    `steps=${s.envSteps} ep=${s.episodesDone} eps=${s.epsilon.toFixed(2)} succ50=${(s.successRate * 100).toFixed(0)}% ` +
      `avgR=${s.avgReward.toFixed(2)} tcomp=${s.avgCompletionTime.toFixed(1)}s loss=${s.loss.toFixed(3)} ` +
      `(${((performance.now() - t0) / 1000).toFixed(0)}s)`,
  );
}
const after = await evaluate('trained', 200);
console.log(`\nRESULT untrained=${(before * 100).toFixed(0)}% trained=${(after * 100).toFixed(0)}%`);
process.exit(after > before + 0.3 ? 0 : 1);
