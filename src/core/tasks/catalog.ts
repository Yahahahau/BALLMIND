import type { TaskInfo } from '../task/types';
import { obbyInfo } from './obby/info';
import { reachSquareInfo } from './reachSquare/info';

/** Task list for the UI (no physics code). Add new tasks here AND in registry.ts. */
export const TASK_CATALOG: TaskInfo[] = [reachSquareInfo, obbyInfo];
