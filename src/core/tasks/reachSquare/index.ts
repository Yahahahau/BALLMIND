import type { TaskDefinition } from '../../task/types';
import { createReachFpEnvironment, createReachSquareEnvironment, createReachVisionEnvironment } from './environment';
import { reachFpInfo, reachSquareInfo, reachVisionInfo } from './info';

export const reachSquareTask: TaskDefinition = {
  info: reachSquareInfo,
  createEnvironment: createReachSquareEnvironment,
};

/**
 * Vision variant. The network is a CNN + dense head, trained by the same Double DQN loop from real
 * Rapier outcomes. Heavier per update than the state DQN, hence fewer updates per decision and a
 * smaller replay (frames are stored as uint8).
 */
export const reachVisionTask: TaskDefinition = {
  info: reachVisionInfo,
  vision: reachVisionInfo.vision,
  dqn: {
    hidden: [64],
    learningRate: 0.0005,
    batchSize: 32,
    bufferCapacity: 12000,
    minReplay: 1500,
    targetSyncEvery: 400,
    epsDecaySteps: 25000,
    updatesPerStep: 0.25,
  },
  createEnvironment: createReachVisionEnvironment,
};

/**
 * Vision Agent V2 (first-person, gaze actions, CNN + GRU). A separate task/model slot: the sensor-based
 * Reach agent and the v1 bird's-eye vision agent are untouched and keep their saved models.
 */
export const reachFpTask: TaskDefinition = {
  info: reachFpInfo,
  vision: reachFpInfo.vision,
  dqn: {
    hidden: [64],
    learningRate: 0.0004,
    batchSize: 8, // sequences per update
    bufferCapacity: 12000,
    minReplay: 1500,
    targetSyncEvery: 300,
    epsDecaySteps: 30000,
    updatesPerStep: 0.12,
  },
  createEnvironment: createReachFpEnvironment,
};
