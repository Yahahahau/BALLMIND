import type { ActionSet } from '../action/actionSet';

export interface StepResult {
  reward: number;
  /** Episode ended (success, failure or time limit). */
  done: boolean;
  /** Real terminal state – the value of the next state is 0. False for time-limit truncation. */
  terminal: boolean;
  success: boolean;
  failed: boolean;
  /** The ball fell and was put back at the last checkpoint (the episode continues). */
  respawned?: boolean;
  /** Observation after the step (buffer is reused by the environment – copy it). */
  obs: Float32Array;
}

export interface BodyPose {
  p: [number, number, number];
  q: [number, number, number, number];
}

/** Everything the renderer needs to draw the current moment. */
export interface EnvSnapshot {
  /** Increments whenever the world is teleported/reset – renderers snap instead of interpolating. */
  epoch: number;
  bodies: Record<string, BodyPose>;
  /** Flat [x,y,z,qx,qy,qz,qw]* poses of kinematic bodies (moving platforms, hazards) in a fixed order. */
  dyn?: Float32Array;
  /** Free-form scalar info (grounded flag, distance …). */
  info: Record<string, number>;
}

/** Static description handed to the UI/renderer when a task is loaded. */
export interface TaskInfo {
  id: string;
  title: string;
  description: string;
  observationSize: number;
  actionLabels: string[];
  /** Task-specific geometry the renderer needs (arena size, target size…). */
  geometry: Record<string, number>;
  /** Vision task: the AI sees camera images instead of coordinates. */
  vision?: { scene: string; proprioSize: number; defaults?: Partial<import('../vision/config').VisionSettings> };
}

/** Optional per-episode details a task can report (used for per-stage statistics). */
export interface EpisodeSummary {
  stage: number;
  /** Best path progress reached, 0..1. */
  progress: number;
  checkpoints: number;
  falls: number;
  /** Started from a chosen checkpoint: excluded from stage success statistics. */
  partial: boolean;
}

export interface TaskNotice {
  kind: 'unlock' | 'info';
  text: string;
  stage?: number;
}

/** Compact recording of one episode for the Training Replay viewer. */
export interface ReplayTrace {
  stage: number;
  /** Simulation time (s) at decision 0 – moving platforms are pure functions of time. */
  t0: number;
  startIdx: number;
  /** Ball position per decision, centimetres: x,y,z interleaved. */
  pos: Int16Array;
  /** [decision, kind, value]; kind: 1 land, 2 checkpoint, 3 fall, 4 finish. */
  events: number[][];
}

export interface Environment {
  readonly observationSize: number;
  readonly actionSet: ActionSet;
  /** Physics sub-steps (1/60 s each) per agent decision. */
  readonly substepsPerDecision: number;
  readonly maxDecisions: number;
  /** Teleport everything to a fresh episode start. Returns the first observation. */
  reset(): Float32Array;
  /** Advance one physics step while applying the chosen action. */
  physicsStep(action: number, firstSubstep: boolean): void;
  /** Called after `substepsPerDecision` physics steps: compute observation, reward and termination. */
  evaluate(decisionIndex: number): StepResult;
  snapshot(): EnvSnapshot;
  dispose(): void;

  // ---- optional multi-stage / replay extensions (Reach The Square implements none of these) ----
  episodeSummary?(): EpisodeSummary;
  /** Extra live numbers for the dashboard (current stage, checkpoint, progress…). */
  liveStats?(): Record<string, unknown>;
  command?(cmd: string, arg: unknown): void;
  exportState?(): unknown;
  importState?(state: unknown): void;
  drainNotices?(): TaskNotice[];
  /** Trace of the episode that just ended (call once after `done`). */
  takeTrace?(): ReplayTrace | null;
}

/**
 * A training task bundles its own environment, observation, action and reward
 * definitions. Add a new task by implementing this interface and registering it.
 */
export interface TaskDefinition {
  info: TaskInfo;
  /** Task-specific DQN overrides (network size, buffer, update ratio…). */
  dqn?: Partial<import('../rl/dqn').DQNConfig>;
  /** Set for tasks where the AI perceives the world through its camera. */
  vision?: import('../vision/config').VisionTaskSpec;
  createEnvironment(seed: number): Promise<Environment>;
}
