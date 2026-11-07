import type { DQNConfig } from '../rl/dqn';

export interface EpisodeRecord {
  /** 1-based episode number. */
  ep: number;
  reward: number;
  success: boolean;
  /** Decisions taken in the episode. */
  steps: number;
}

export interface Counters {
  episodes: number;
  envSteps: number;
  successes: number;
  totalScore: number;
}

/** Everything needed to resume training (replay buffer excluded – it refills in seconds). */
export interface ModelBlob {
  version: 1;
  taskId: string;
  obsSize: number;
  actionCount: number;
  config: DQNConfig;
  online: Float32Array;
  target: Float32Array;
  adamM: Float32Array;
  adamV: Float32Array;
  adamT: number;
  trainSteps: number;
  epsSteps: number;
  counters: Counters;
  history: { reward: Float32Array; success: Uint8Array; steps: Uint16Array };
  savedAt: number;
}

export interface TrainingStats {
  episode: number;
  episodesDone: number;
  envSteps: number;
  trainSteps: number;
  totalScore: number;
  successes: number;
  /** Rolling success rate (last 50 episodes), 0..1. */
  successRate: number;
  currentReward: number;
  avgReward: number;
  /** Mean simulated seconds per successful episode (rolling). */
  avgCompletionTime: number;
  epsilon: number;
  loss: number;
  bufferSize: number;
  demo: boolean;
  demoEpisodes: number;
  demoSuccessRate: number;
}

// ---------- JSON import/export (typed arrays <-> base64) ----------

function toB64(bytes: Uint8Array): string {
  let s = '';
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) s += String.fromCharCode(...bytes.subarray(i, i + chunk));
  return btoa(s);
}

function fromB64(str: string): Uint8Array {
  const s = atob(str);
  const out = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) out[i] = s.charCodeAt(i);
  return out;
}

const f32 = (a: Float32Array) => toB64(new Uint8Array(a.buffer, a.byteOffset, a.byteLength));
const unf32 = (s: string) => {
  const b = fromB64(s);
  return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4);
};

export function modelToJson(m: ModelBlob): string {
  return JSON.stringify({
    format: 'ballmind-model',
    version: m.version,
    taskId: m.taskId,
    obsSize: m.obsSize,
    actionCount: m.actionCount,
    config: m.config,
    online: f32(m.online),
    target: f32(m.target),
    adamM: f32(m.adamM),
    adamV: f32(m.adamV),
    adamT: m.adamT,
    trainSteps: m.trainSteps,
    epsSteps: m.epsSteps,
    counters: m.counters,
    history: {
      reward: f32(m.history.reward),
      success: toB64(m.history.success),
      steps: toB64(new Uint8Array(m.history.steps.buffer, m.history.steps.byteOffset, m.history.steps.byteLength)),
    },
    savedAt: m.savedAt,
  });
}

export function modelFromJson(text: string): ModelBlob {
  const j = JSON.parse(text);
  if (j.format !== 'ballmind-model' || j.version !== 1) throw new Error('Not a BALLMIND model file');
  const stepsBytes = fromB64(j.history.steps);
  const steps = new Uint16Array(stepsBytes.buffer, stepsBytes.byteOffset, stepsBytes.byteLength / 2);
  return {
    version: 1,
    taskId: j.taskId,
    obsSize: j.obsSize,
    actionCount: j.actionCount,
    config: j.config,
    online: unf32(j.online),
    target: unf32(j.target),
    adamM: unf32(j.adamM),
    adamV: unf32(j.adamV),
    adamT: j.adamT,
    trainSteps: j.trainSteps,
    epsSteps: j.epsSteps,
    counters: j.counters,
    history: { reward: unf32(j.history.reward), success: fromB64(j.history.success), steps },
    savedAt: j.savedAt,
  };
}
