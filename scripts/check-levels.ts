import { STAGES } from '../src/core/tasks/obby/levels/compile';
// Geometry sanity (test-only): edge-to-edge gap and height step between consecutive platforms.
for (const st of STAGES) {
  const P = st.def.platforms;
  const worst: string[] = [];
  for (let i = 1; i < P.length; i++) {
    const a = P[i - 1], b = P[i];
    const d = Math.hypot(b.pos[0] - a.pos[0], b.pos[2] - a.pos[2]);
    const half = (p: typeof a) => Math.max(p.size[0], p.size[1]) / 2;
    const gap = d - (a.size[1] / 2 + b.size[1] / 2);
    const dy = b.pos[1] - a.pos[1];
    if (gap > 3.6 || dy > 0.7) worst.push(`#${i} gap~${gap.toFixed(1)} dy=${dy.toFixed(1)} ${b.kind}${b.motion ? '+mv' : ''}${b.spin ? '+spin' : ''}`);
  }
  console.log(`S${st.def.id} ${st.def.name}: ${P.length} plats, tough links: ${worst.join(' | ') || 'none'}`);
}
