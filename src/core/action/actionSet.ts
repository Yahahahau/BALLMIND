/**
 * Action System. An action is a data record, not hard-coded logic: new abilities
 * (e.g. "spin", "dash") are added by appending to the list and teaching the
 * controller how to interpret a new field. The DQN only sees `size`.
 */
export interface ActionDef {
  id: string;
  label: string;
  /** Desired horizontal push direction in world space [x, z]. Forward = -z, right = +x. */
  move: [number, number];
  jump: boolean;
  /** Actively damp linear + angular velocity. */
  brake: boolean;
}

export class ActionSet {
  constructor(readonly actions: ActionDef[]) {}

  get size(): number {
    return this.actions.length;
  }

  get(i: number): ActionDef {
    return this.actions[i];
  }
}

const D = Math.SQRT1_2;

export function createLocomotionActions(): ActionSet {
  const a = (id: string, label: string, mx: number, mz: number, jump = false, brake = false): ActionDef => ({
    id,
    label,
    move: [mx, mz],
    jump,
    brake,
  });
  return new ActionSet([
    a('stop', 'Stop', 0, 0, false, true),
    a('forward', 'Forward', 0, -1),
    a('backward', 'Backward', 0, 1),
    a('left', 'Left', -1, 0),
    a('right', 'Right', 1, 0),
    a('forward_left', 'Forward-Left', -D, -D),
    a('forward_right', 'Forward-Right', D, -D),
    a('backward_left', 'Backward-Left', -D, D),
    a('backward_right', 'Backward-Right', D, D),
    a('jump', 'Jump', 0, 0, true),
    a('jump_forward', 'Jump Forward', 0, -1, true),
    a('jump_backward', 'Jump Backward', 0, 1, true),
    a('jump_left', 'Jump Left', -1, 0, true),
    a('jump_right', 'Jump Right', 1, 0, true),
  ]);
}
