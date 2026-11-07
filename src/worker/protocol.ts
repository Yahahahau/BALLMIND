import type { EnvSnapshot, TaskInfo, TaskNotice } from '../core/task/types';
import type { EpisodeRecord, ModelBlob, ReplayRecord, TrainingStats } from '../core/training/model';

/** Main thread -> simulation worker. */
export type ToWorker =
  | { type: 'init'; taskId: string; seed: number; model?: ModelBlob }
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
  | { type: 'load'; model: ModelBlob };

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
  | { type: 'error'; message: string };
