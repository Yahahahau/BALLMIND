import type { TaskDefinition } from '../../task/types';
import { createObbyEnvironment } from './environment';
import { obbyInfo } from './info';

export const obbyTask: TaskDefinition = {
  info: obbyInfo,
  createEnvironment: createObbyEnvironment,
  // Bigger observation (83 inputs) and a long multi-stage task: a wider net, a larger buffer
  // and ~1 gradient update per 2 decisions (the network update is the CPU bottleneck on an iPad).
  dqn: {
    hidden: [96, 96],
    bufferCapacity: 60000,
    minReplay: 2000,
    epsDecaySteps: 40000,
    epsEnd: 0.06,
    gamma: 0.98,
    targetSyncEvery: 800,
    updatesPerStep: 0.5,
  },
};
