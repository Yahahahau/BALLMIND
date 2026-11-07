import type { ActionSet } from '../action/actionSet';

export interface StepResult {
  reward: number;
  /** Episode ended (success, failure or time limit). */
  done: boolean;
  /** Real terminal state – the value of the next state is 0. False for time-limit truncation. */
  terminal: boolean;
  success: boolean;
  failed: boolean;
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
}

/**
 * A training task bundles its own environment, observation, action and reward
 * definitions. Add a new task by implementing this interface and registering it.
 */
export interface TaskDefinition {
  info: TaskInfo;
  createEnvironment(seed: number): Promise<Environment>;
}
