import type { TaskDefinition } from '../../task/types';
import { createReachSquareEnvironment } from './environment';
import { reachSquareInfo } from './info';

export const reachSquareTask: TaskDefinition = {
  info: reachSquareInfo,
  createEnvironment: createReachSquareEnvironment,
};
