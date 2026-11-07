import { getTask } from '../src/core/tasks/registry';
import { TrainingManager } from '../src/core/training/trainingManager';
import { modelFromJson, modelToJson } from '../src/core/training/model';

function ok(c: boolean, m: string) { if (!c) { console.error('FAIL', m); process.exit(1); } console.log('PASS', m); }

for (const id of ['reach-square', 'obby-challenge']) {
  const a = await TrainingManager.create(getTask(id), 3, { minReplay: 300 });
  a.runDecisions(id === 'reach-square' ? 4000 : 2500);
  const s = a.stats();
  ok(s.episodesDone > 0 && s.trainSteps > 0, `${id}: trained (${s.episodesDone} eps, ${s.trainSteps} updates)`);
  const blob = a.exportModel(true);
  ok(!!blob.replay && blob.replay.n === a.agent.replay.size, `${id}: replay dumped (${blob.replay!.n})`);
  const json = modelToJson({ ...blob, replay: undefined });
  const back = modelFromJson(json);
  const b = await TrainingManager.create(getTask(id), 9, { minReplay: 300 });
  b.importModel({ ...back, replay: blob.replay });
  ok(b.stats().episodesDone === s.episodesDone && b.agent.replay.size === a.agent.replay.size, `${id}: import restores counters+buffer`);
  ok(Math.abs(b.agent.online.params[5] - a.agent.online.params[5]) < 1e-9, `${id}: weights equal`);
  if (id !== 'reach-square') {
    ok((s.extra?.stages as unknown[]).length === 5, 'obby: 5 per-stage stats');
    ok(a.drainTraces().length > 0, 'obby: traces emitted');
    try { (await TrainingManager.create(getTask('reach-square'), 1)).importModel(back); ok(false, 'cross-task import must fail'); } catch (e) { ok(true, 'cross-task import rejected: ' + (e as Error).message); }
  }
}
console.log('all training tests passed');
