import type { TaskInfo } from '../../task/types';
import { createObbyActions } from './actions';
import { OBS_SIZE } from './observation';

export const obbyInfo: TaskInfo = {
  id: 'obby-challenge',
  title: 'OBBY CHALLENGE',
  description:
    'Five obstacle courses of rising difficulty: gaps, narrow paths, moving and spinning platforms. One brain learns them all and must reach every finish line using only real physics.',
  observationSize: OBS_SIZE,
  actionLabels: createObbyActions().actions.map((a) => a.label),
  geometry: { stages: 5 },
};
