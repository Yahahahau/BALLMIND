/**
 * Simulation worker: Physics + AI live here, completely off the UI/render thread,
 * so training can never freeze the interface or stutter the 3D view.
 */
import { getTask } from '../core/tasks/registry';
import { TrainingManager } from '../core/training/trainingManager';
import type { FromWorker, ToWorker } from './protocol';

interface WorkerScope {
  postMessage(msg: unknown, transfer?: Transferable[]): void;
  onmessage: ((e: MessageEvent<ToWorker>) => void) | null;
}
const ctx = self as unknown as WorkerScope;

const TICK_MS = 16;
/** Wall-clock budget per tick for simulation + learning (the rest is left for the system). */
const BUDGET_MS = 11;

let mgr: TrainingManager | null = null;
let last = 0;
let sentOnce = false;

function post(msg: FromWorker, transfer?: Transferable[]): void {
  ctx.postMessage(msg, transfer);
}

function frame(): void {
  if (!mgr) return;
  const t = performance.now();
  const dt = Math.min(0.05, (t - last) / 1000);
  last = t;
  mgr.advance(dt, BUDGET_MS);
  const episodes = mgr.drainEpisodes();
  if (mgr.dirty || episodes.length || !sentOnce) {
    sentOnce = true;
    mgr.dirty = false;
    post({
      type: 'frame',
      snap: mgr.snapshot(),
      stats: mgr.stats(),
      episodes,
      state: mgr.state,
      dps: mgr.decisionsPerSecond,
    });
  }
}

ctx.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === 'init') {
      mgr?.env.dispose();
      mgr = await TrainingManager.create(getTask(m.taskId), m.seed);
      let restored = false;
      if (m.model) {
        try {
          mgr.importModel(m.model);
          restored = true;
        } catch (err) {
          post({ type: 'error', message: `Saved model ignored: ${(err as Error).message}` });
        }
      }
      last = performance.now();
      post({ type: 'ready', info: mgr.task.info, stats: mgr.stats(), history: mgr.fullHistory(), restored });
      setInterval(frame, TICK_MS);
      return;
    }
    if (!mgr) return;
    switch (m.type) {
      case 'start':
        mgr.start();
        break;
      case 'pause':
        mgr.pause();
        break;
      case 'resume':
        mgr.resume();
        break;
      case 'speed':
        mgr.setSpeed(m.value);
        break;
      case 'resetEnv':
        mgr.resetEnvironment();
        break;
      case 'resetAI':
        mgr.resetAI();
        post({ type: 'reset-ai', stats: mgr.stats() });
        break;
      case 'demo':
        mgr.setDemo(m.on);
        break;
      case 'save': {
        const model = mgr.exportModel();
        post({ type: 'saved', model, reason: m.reason });
        break;
      }
      case 'export':
        post({ type: 'exported', model: mgr.exportModel() });
        break;
      case 'load':
        mgr.importModel(m.model);
        post({ type: 'loaded', stats: mgr.stats(), history: mgr.fullHistory() });
        break;
    }
    mgr.dirty = true;
  } catch (err) {
    post({ type: 'error', message: (err as Error).message });
  }
};
