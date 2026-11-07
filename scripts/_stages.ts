import { STAGES } from '../src/core/tasks/obby/levels/compile';
for (const s of STAGES) {
  const d = s.def;
  const gaps: number[] = [];
  for (let i = 1; i < d.platforms.length; i++) {
    const a = d.platforms[i - 1], b = d.platforms[i];
    gaps.push(Math.round(Math.hypot(b.pos[0] - a.pos[0], b.pos[2] - a.pos[2]) * 10) / 10);
  }
  console.log(`S${d.id} ${d.name}: ${d.platforms.length} platforms, ${d.hazards.length} hazards, path ${s.total.toFixed(0)} m, checkpoints ${s.checkpointIdx.join(',')}, voidY ${s.voidY.toFixed(1)}`);
  console.log('  bounds', s.bounds.min.map((v) => v.toFixed(0)).join(','), '->', s.bounds.max.map((v) => v.toFixed(0)).join(','));
}
