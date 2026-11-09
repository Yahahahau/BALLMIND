import { createLocomotionActions } from '../../action/actionSet';
import type { TaskInfo } from '../../task/types';
import { REACH_CONFIG as C } from './config';
import { OBSERVATION_SIZE, PROPRIO_SIZE } from './observation';

/** Light-weight task description (no physics imports) – safe to bundle into the UI thread. */
export const reachSquareInfo: TaskInfo = {
  id: 'reach-square',
  title: 'REACH THE SQUARE',
  description:
    'The AI steers the ball across the floor and must touch the target square – without rolling off the edge. The square moves to a new random place every episode.',
  observationSize: OBSERVATION_SIZE,
  actionLabels: createLocomotionActions().actions.map((a) => a.label),
  geometry: {
    arenaHalf: C.arenaHalf,
    floorThickness: C.floorThickness,
    ballRadius: C.ballRadius,
    targetHalf: C.targetHalf,
    startX: C.start[0],
    startZ: C.start[2],
  },
};

export const reachVisionInfo: TaskInfo = {
  id: 'reach-vision',
  title: 'REACH THE SQUARE · VISION',
  description:
    'Same goal as Reach The Square, but the AI is blind to coordinates: it only gets camera pictures from its own AI camera plus its physics sensors, and must learn to see the target with a CNN.',
  observationSize: PROPRIO_SIZE,
  actionLabels: createLocomotionActions().actions.map((a) => a.label),
  geometry: reachSquareInfo.geometry,
  vision: { scene: 'reach', proprioSize: PROPRIO_SIZE },
};
