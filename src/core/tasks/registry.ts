import type { TaskDefinition } from '../task/types';
import { obbyTask } from './obby';
import { reachFpTask, reachSquareTask, reachVisionTask } from './reachSquare';

/** Worker/test-side registry (full definitions incl. physics). Keep in sync with catalog.ts. */
export const TASKS: TaskDefinition[] = [reachSquareTask, reachVisionTask, reachFpTask, obbyTask];

export function getTask(id: string): TaskDefinition {
  const t = TASKS.find((x) => x.info.id === id);
  if (!t) throw new Error(`Unknown task: ${id}`);
  return t;
}
