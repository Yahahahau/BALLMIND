export const REACH_CONFIG = {
  /** Half extent of the square floor (metres). */
  arenaHalf: 6,
  floorThickness: 1,
  ballRadius: 0.5,
  start: [0, 0.5, 0] as [number, number, number],
  /** Half side of the target tile. */
  targetHalf: 0.9,
  /** Height of the visible target tile. */
  tileHeight: 0.06,
  /** Target spawn bounds (inside the arena, keeping a safety margin from the edge). */
  spawnHalf: 4.6,
  /** Minimum distance between ball start and target. */
  minSpawnDistance: 2.6,
  substepsPerDecision: 6,
  maxDecisions: 200,
  /** Ball centre below this height means it has fallen off the arena. */
  fallY: -1.0,
  vMax: 10,
};
