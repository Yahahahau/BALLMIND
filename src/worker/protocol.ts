import type { EnvSnapshot, TaskInfo, TaskNotice } from '../core/task/types';
import type { VisionSettings } from '../core/vision/config';
import type { EpisodeRecord, ModelBlob, ReplayRecord, TrainingStats } from '../core/training/model';

/** Main thread -> simulation worker. */
export type ToWorker =
  | { type: 'init'; taskId: string; seed: number; model?: ModelBlob; vision?: Partial<VisionSettings> }
  | { type: 'start' }
  | { type: 'pause' }
  | { type: 'resume' }
  | { type: 'speed'; value: number }
  | { type: 'stop' }
  | { type: 'command'; cmd: string; arg: unknown }
  | { type: 'resetEnv' }
  | { type: 'resetAI' }
  | { type: 'demo'; on: boolean }
  | { type: 'save'; reason: string }
  | { type: 'export' }
  | { type: 'load'; model: ModelBlob }
  /** Stream the AI camera image to the UI (off by default; ~5 Hz). */
  | { type: 'visionPreview'; on: boolean };

export type RunStateName = 'idle' | 'running' | 'paused';

/** Simulation worker -> main thread. */
export type FromWorker =
  | { type: 'ready'; info: TaskInfo; stats: TrainingStats; history: EpisodeRecord[]; loss: number[]; restored: boolean }
  | {
      type: 'frame';
      snap: EnvSnapshot & { action: number };
      stats: TrainingStats;
      episodes: EpisodeRecord[];
      loss: number[];
      traces: ReplayRecord[];
      notices: TaskNotice[];
      state: RunStateName;
      dps: number;
    }
  | { type: 'saved'; model: ModelBlob; reason: string }
  | { type: 'exported'; model: ModelBlob }
  | { type: 'loaded'; stats: TrainingStats; history: EpisodeRecord[]; loss: number[] }
  | { type: 'reset-ai'; stats: TrainingStats }
  | { type: 'visionFrame'; size: number; rgb: Uint8Array }
  | { type: 'visionInfo'; settings: VisionSettings; kind: string; note?: string; params: number; macs: number }
  | { type: 'error'; message: string };
