import type { DQNConfig } from '../rl/dqn';
import type { ReplayDump } from '../rl/replayBuffer';
import type { ReplayTrace } from '../task/types';

export const MODEL_VERSION = 2;

export interface EpisodeRecord {
  /** 1-based episode number. */
  ep: number;
  reward: number;
  success: boolean;
  /** Decisions taken in the episode. */
  steps: number;
  // multi-stage tasks only
  stage?: number;
  progress?: number;
  checkpoints?: number;
  falls?: number;
  /** Started from a chosen checkpoint: excluded from stage statistics. */
  partial?: boolean;
}

export interface Counters {
  episodes: number;
  envSteps: number;
  successes: number;
  totalScore: number;
}

export interface StageTotals {
  episodes: number;
  successes: number;
  falls: number;
  bestTime: number;
  bestProgress: number;
}

export interface HistoryArrays {
  reward: Float32Array;
  success: Uint8Array;
  steps: Uint16Array;
  stage?: Uint8Array;
  progress?: Float32Array;
  checkpoints?: Uint8Array;
  falls?: Uint8Array;
  partial?: Uint8Array;
}

/** Everything needed to resume training. v1 files (Reach The Square only) are still accepted. */
export interface ModelBlob {
  version: number;
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
  history: HistoryArrays;
  lossHistory?: Float32Array;
  stageTotals?: StageTotals[];
  /** Task-specific state (curriculum, start mode…). */
  taskState?: unknown;
  /** Network architecture fingerprint (absent in older files). */
  arch?: string;
  /** Present in IndexedDB saves, omitted from exported files. */
  replay?: ReplayDump;
  savedAt: number;
}

export interface StageStats {
  episodes: number;
  attempts: number;
  successes: number;
  falls: number;
  successRate: number;
  avgReward: number;
  avgTime: number;
  bestTime: number;
  avgProgress: number;
  avgCheckpoints: number;
  bestProgress: number;
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
  /** Task-specific live values (Obby: stage, checkpoint, progress, curriculum…) and per-stage stats. */
  extra?: Record<string, unknown> & { stages?: StageStats[] };
  /** Present for vision tasks. */
  vision?: { size: number; stack: number; params: number; renderMs: number; kind: string; mode: string; yaw: number; pitch: number; fov: number; gazeAction: string; frames: number; decisionsPerSecond: number };
}

/** One recorded episode for the Training Replay viewer. */
export interface ReplayRecord {
  ep: number;
  stage: number;
  reward: number;
  success: boolean;
  progress: number;
  checkpoints: number;
  falls: number;
  /** Why it was kept: first / milestone / periodic / recent. */
  kind: 'first' | 'milestone' | 'periodic' | 'recent';
  trace: ReplayTrace;
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
const bytes = (a: ArrayBufferView) => new Uint8Array(a.buffer, a.byteOffset, a.byteLength);
const f32 = (a: Float32Array) => toB64(bytes(a));
const unf32 = (s: string) => {
  const b = fromB64(s);
  return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4);
};
const u16 = (a: Uint16Array) => toB64(bytes(a));
const unu16 = (s: string) => {
  const b = fromB64(s);
  return new Uint16Array(b.buffer, b.byteOffset, b.byteLength / 2);
};

export function modelToJson(m: ModelBlob): string {
  const h = m.history;
  return JSON.stringify({
    format: 'ballmind-model',
    version: MODEL_VERSION,
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
      reward: f32(h.reward),
      success: toB64(h.success),
      steps: u16(h.steps),
      stage: h.stage ? toB64(h.stage) : undefined,
      progress: h.progress ? f32(h.progress) : undefined,
      checkpoints: h.checkpoints ? toB64(h.checkpoints) : undefined,
      falls: h.falls ? toB64(h.falls) : undefined,
      partial: h.partial ? toB64(h.partial) : undefined,
    },
    lossHistory: m.lossHistory ? f32(m.lossHistory) : undefined,
    stageTotals: m.stageTotals,
    taskState: m.taskState,
    savedAt: m.savedAt,
    arch: m.arch,
  });
}

/** Parses and validates an exported model file. Throws a readable error if it is not compatible. */
export function modelFromJson(text: string): ModelBlob {
  let j: any;
  try {
    j = JSON.parse(text);
  } catch {
    throw new Error('File is not valid JSON');
  }
  if (j?.format !== 'ballmind-model') throw new Error('Not a BALLMIND model file');
  if (typeof j.version !== 'number' || j.version < 1 || j.version > MODEL_VERSION)
    throw new Error(`Model version ${j.version} is not supported (this app reads versions 1–${MODEL_VERSION})`);
  const h = j.history ?? {};
  const optU8 = (s?: string) => (s ? fromB64(s) : undefined);
  return {
    version: j.version,
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
    history: {
      reward: unf32(h.reward),
      success: fromB64(h.success),
      steps: unu16(h.steps),
      stage: optU8(h.stage),
      progress: h.progress ? unf32(h.progress) : undefined,
      checkpoints: optU8(h.checkpoints),
      falls: optU8(h.falls),
      partial: optU8(h.partial),
    },
    lossHistory: j.lossHistory ? unf32(j.lossHistory) : undefined,
    stageTotals: j.stageTotals,
    taskState: j.taskState,
    savedAt: j.savedAt,
    arch: typeof j.arch === 'string' ? j.arch : undefined,
  };
}
