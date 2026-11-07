import { ActionSet, createLocomotionActions } from '../../action/actionSet';

/** 15 discrete actions: 8 directions, jump (+4 directions), brake and neutral. */
export function createObbyActions(): ActionSet {
  const base = createLocomotionActions().actions; // stop(=brake), 8 moves, jump, 4 jump-moves
  return new ActionSet([
    ...base.map((a) => (a.id === 'stop' ? { ...a, id: 'brake', label: 'Brake' } : a)),
    { id: 'neutral', label: 'Neutral', move: [0, 0], jump: false, brake: false },
  ]);
}
