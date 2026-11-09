import type { AgentState, DQNConfig } from './dqn';
import type { ReplayDump } from './replayBuffer';

/** What the TrainingManager needs from an agent – implemented by the state DQN and the vision DQN. */
export interface Learner {
  config: DQNConfig;
  readonly trainSteps: number;
  readonly lastLoss: number;
  readonly epsilon: number;
  readonly paramCount: number;
  readonly replay: { readonly size: number; dump(): ReplayDump | undefined };
  /** Identifies the network architecture (for model compatibility checks). */
  arch(): string;
  act(obs: Float32Array, explore: boolean): number;
  remember(obs: Float32Array, action: number, reward: number, next: Float32Array, terminal: boolean): void;
  train(): number | null;
  boostExploration(eps: number): void;
  reset(seed?: number): void;
  exportState(): AgentState;
  importState(s: AgentState): void;
  loadReplay(d: ReplayDump): void;
}
