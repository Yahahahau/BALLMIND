/**
 * Render-side interpolation of authoritative physics snapshots (DOM/three-free, unit-testable).
 *
 * The worker stamps every snapshot with the physics time (`simTime`, fixed 1/60 s steps). The renderer keeps
 * a short buffer and draws the world at a *render time* that runs slightly behind the newest snapshot, so
 * every displayed frame lies between two real physics states. This replaces exponential "chase" smoothing
 * (which turns irregular snapshot arrival into visible stutter) and never touches the physics state.
 */
export interface Sample {
  t: number; // physics time (s)
  p: [number, number, number];
  q: [number, number, number, number];
  target?: [number, number, number];
  yaw?: number; // deg (first-person gaze)
  pitch?: number;
  epoch: number;
  dyn?: Float32Array;
}

export interface Pose {
  p: [number, number, number];
  q: [number, number, number, number];
  target?: [number, number, number];
  yaw?: number;
  pitch?: number;
  dyn?: Float32Array;
  /** render time lag behind the newest snapshot, seconds of physics time */
  lag: number;
}

const wrap = (d: number) => ((((d + 180) % 360) + 360) % 360) - 180;

export class SnapshotInterpolator {
  private buf: { s: Sample; wall: number }[] = [];
  private rt = 0;
  private rate = 1; // physics seconds per wall second
  private lastWall = 0;
  private started = false;
  /** diagnostics */
  stats = { pushed: 0, underruns: 0, rate: 1, lag: 0 };

  clear(): void {
    this.buf.length = 0;
    this.started = false;
  }

  get ready(): boolean {
    return this.buf.length > 0;
  }

  push(s: Sample, wallMs: number): void {
    const last = this.buf[this.buf.length - 1];
    if (last && s.epoch !== last.s.epoch) this.clear();
    const l2 = this.buf[this.buf.length - 1];
    if (l2 && s.t <= l2.s.t) return; // duplicate / out of order
    this.buf.push({ s, wall: wallMs });
    if (this.buf.length > 12) this.buf.shift();
    this.stats.pushed++;
    if (this.buf.length >= 3) {
      const a = this.buf[Math.max(0, this.buf.length - 8)];
      const b = this.buf[this.buf.length - 1];
      const dw = (b.wall - a.wall) / 1000;
      if (dw > 0.02) {
        const r = (b.s.t - a.s.t) / dw;
        this.rate += (r - this.rate) * 0.3;
        this.stats.rate = this.rate;
      }
    }
    if (!this.started) {
      this.started = true;
      this.rt = s.t;
      this.lastWall = wallMs;
    }
  }

  /** Pose to draw at wall-clock time `wallMs`. */
  sample(wallMs: number): Pose | null {
    const n = this.buf.length;
    if (!n) return null;
    const newest = this.buf[n - 1];
    const dt = Math.max(0, Math.min(0.1, (wallMs - this.lastWall) / 1000));
    this.lastWall = wallMs;
    // desired lag: ~1.5 snapshot intervals (physics time); drops to 0 once snapshots stop (paused)
    let spacing = 1 / 60;
    if (n >= 2) spacing = Math.max(1 / 120, (newest.s.t - this.buf[Math.max(0, n - 6)].s.t) / Math.min(5, n - 1));
    const sinceWall = (wallMs - newest.wall) / 1000;
    const hold = Math.max(0, 1 - sinceWall / 0.12);
    const target = newest.s.t - spacing * 1.5 * hold;
    this.rt += dt * this.rate * hold + (target - this.rt) * Math.min(1, dt * 6);
    if (this.rt > newest.s.t) {
      this.rt = newest.s.t;
      this.stats.underruns++;
    }
    const oldest = this.buf[0].s.t;
    if (this.rt < oldest) this.rt = oldest;
    // bracket
    let i = n - 1;
    while (i > 0 && this.buf[i - 1].s.t > this.rt) i--;
    const b = this.buf[i].s;
    const a = i > 0 ? this.buf[i - 1].s : b;
    const span = b.t - a.t;
    const k = span > 1e-9 ? Math.min(1, Math.max(0, (this.rt - a.t) / span)) : 1;
    const out: Pose = {
      p: [a.p[0] + (b.p[0] - a.p[0]) * k, a.p[1] + (b.p[1] - a.p[1]) * k, a.p[2] + (b.p[2] - a.p[2]) * k],
      q: nlerp(a.q, b.q, k),
      lag: newest.s.t - this.rt,
      dyn: a.dyn ?? b.dyn,
    };
    if (a.target && b.target) out.target = [a.target[0] + (b.target[0] - a.target[0]) * k, a.target[1] + (b.target[1] - a.target[1]) * k, a.target[2] + (b.target[2] - a.target[2]) * k];
    if (a.yaw !== undefined && b.yaw !== undefined) {
      out.yaw = wrap(a.yaw + wrap(b.yaw - a.yaw) * k);
      out.pitch = a.pitch! + (b.pitch! - a.pitch!) * k;
    }
    this.stats.lag = out.lag;
    return out;
  }
}

function nlerp(a: [number, number, number, number], b: [number, number, number, number], k: number): [number, number, number, number] {
  const d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
  const s = d < 0 ? -1 : 1;
  const x = a[0] + (s * b[0] - a[0]) * k;
  const y = a[1] + (s * b[1] - a[1]) * k;
  const z = a[2] + (s * b[2] - a[2]) * k;
  const w = a[3] + (s * b[3] - a[3]) * k;
  const l = Math.hypot(x, y, z, w) || 1;
  return [x / l, y / l, z / l, w / l];
}
