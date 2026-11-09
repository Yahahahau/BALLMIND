/**
 * Simulation worker: Physics + AI live here, completely off the UI/render thread,
 * so training can never freeze the interface or stutter the 3D view.
 */
import { getTask } from '../core/tasks/registry';
import { TrainingManager } from '../core/training/trainingManager';
import { createVisionRig } from './gpuVisionRig';
import { sanitizeVision } from '../core/vision/config';
import type { VisionRig } from '../core/vision/rig';
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
let rig: VisionRig | null = null;
let timer: ReturnType<typeof setInterval> | null = null;
let previewOn = false;
let lastPreview = 0;

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
  const traces = mgr.drainTraces();
  const notices = mgr.drainNotices();
  const loss = mgr.drainLoss();
  if (previewOn && mgr.vision && t - lastPreview > 200) {
    lastPreview = t;
    const rgb = mgr.vision.frame.slice();
    post({ type: 'visionFrame', size: mgr.vision.agent.vision.size, rgb }, [rgb.buffer]);
  }
  if (mgr.dirty || episodes.length || traces.length || !sentOnce) {
    sentOnce = true;
    mgr.dirty = false;
    post({
      type: 'frame',
      snap: mgr.snapshot(),
      stats: mgr.stats(),
      episodes,
      loss,
      traces,
      notices,
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
      rig?.dispose();
      rig = null;
      previewOn = false;
      sentOnce = false;
      const task = getTask(m.taskId);
      let note: string | undefined;
      let vopt: { rig: VisionRig; settings: ReturnType<typeof sanitizeVision> } | undefined;
      if (task.vision) {
        const settings = sanitizeVision(m.vision, sanitizeVision(task.vision.defaults));
        const made = createVisionRig(settings);
        rig = made.rig;
        note = made.note;
        vopt = { rig, settings };
      }
      mgr = await TrainingManager.create(task, m.seed, {}, vopt);
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
      if (mgr.vision && vopt) {
        const net = mgr.vision.agent.online;
        post({ type: 'visionInfo', settings: mgr.vision.agent.vision, kind: mgr.vision.rig.kind, note, params: net.params.length, macs: net.macs });
      }
      post({ type: 'ready', info: mgr.task.info, stats: mgr.stats(), history: mgr.fullHistory(), loss: mgr.fullLoss(), restored });
      if (timer) clearInterval(timer);
      timer = setInterval(frame, TICK_MS);
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
      case 'stop':
        mgr.stop();
        break;
      case 'command':
        mgr.command(m.cmd, m.arg);
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
        const model = mgr.exportModel(true);
        post({ type: 'saved', model, reason: m.reason });
        break;
      }
      case 'export':
        post({ type: 'exported', model: mgr.exportModel() });
        break;
      case 'visionPreview':
        previewOn = m.on;
        break;
      case 'load':
        mgr.importModel(m.model);
        post({ type: 'loaded', stats: mgr.stats(), history: mgr.fullHistory(), loss: mgr.fullLoss() });
        break;
    }
    mgr.dirty = true;
  } catch (err) {
    post({ type: 'error', message: (err as Error).message });
  }
};
