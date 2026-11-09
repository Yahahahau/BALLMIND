import type { TaskDefinition } from '../../task/types';
import { createReachSquareEnvironment, createReachVisionEnvironment } from './environment';
import { reachSquareInfo, reachVisionInfo } from './info';

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
