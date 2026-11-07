import type { Vec3 } from '../../../math/quat';
import type { HazardDef, MotionDef, PlatformDef, PlatformKind, SpinDef } from '../types';

const D = Math.PI / 180;

export interface PlatOpts {
  len: number;
  w: number;
  gap?: number;
  dy?: number;
  /** Heading change in degrees (positive = turn left). */
  turn?: number;
  /** Sideways shift in metres (positive = right). */
  lateral?: number;
  thick?: number;
  friction?: number;
  /** [pitch, roll] in degrees. */
  tilt?: [number, number];
  spin?: SpinDef;
}

export interface MoverOpts extends PlatOpts {
  axis: 'forward' | 'right' | 'up';
  amp: number;
  period: number;
  phase?: number;
}

/**
 * Tiny course DSL. A cursor marks the middle of the previous platform's front edge;
 * each call places the next platform ahead of it. Heading 0 = toward -z.
 */
export class CourseBuilder {
  platforms: PlatformDef[] = [];
  hazards: HazardDef[] = [];
  private x = 0;
  private y = 0;
  private z = 0;
  private yaw = 0;

  private fwd(): [number, number] {
    return [-Math.sin(this.yaw), -Math.cos(this.yaw)];
  }
  private right(): [number, number] {
    return [Math.cos(this.yaw), -Math.sin(this.yaw)];
  }

  private place(kind: PlatformKind, o: PlatOpts, extra?: Partial<PlatformDef>, centerShift = 0): PlatformDef {
    if (o.turn) this.yaw += o.turn * D;
    const f = this.fwd();
    const r = this.right();
    const gap = o.gap ?? 0;
    const lat = o.lateral ?? 0;
    const dist = gap + o.len / 2 + centerShift;
    const cx = this.x + f[0] * dist + r[0] * lat;
    const cz = this.z + f[1] * dist + r[1] * lat;
    const cy = this.y + (o.dy ?? 0);
    const p: PlatformDef = {
      kind,
      pos: [cx, cy, cz],
      size: [o.w, o.len],
      yaw: this.yaw,
      thick: o.thick ?? 0.6,
      friction: o.friction ?? (kind === 'slippery' ? 0.04 : 0.9),
      ...(o.tilt ? { tilt: [o.tilt[0] * D, o.tilt[1] * D] as [number, number] } : {}),
      ...(o.spin ? { spin: o.spin } : {}),
      ...extra,
    };
    this.platforms.push(p);
    // cursor -> front edge of this platform (base position)
    this.x = cx + f[0] * (o.len / 2);
    this.z = cz + f[1] * (o.len / 2);
    this.y = cy;
    return p;
  }

  start(len: number, w: number): this {
    // The first platform is centred on the origin, facing -z.
    this.platforms.push({ kind: 'start', pos: [0, 0, 0], size: [w, len], yaw: 0, thick: 0.6, friction: 0.9 });
    this.x = 0;
    this.z = -len / 2;
    this.y = 0;
    return this;
  }
  plat(o: PlatOpts): this {
    this.place('normal', o);
    return this;
  }
  slippery(o: PlatOpts): this {
    this.place('slippery', o);
    return this;
  }
  checkpoint(o: PlatOpts): this {
    this.place('checkpoint', o);
    return this;
  }
  finish(o: PlatOpts): this {
    this.place('finish', o);
    return this;
  }
  spinning(o: PlatOpts & { speed: number; phase?: number }): this {
    this.place('rotating', { ...o, spin: { speed: o.speed, phase: o.phase ?? 0 } });
    return this;
  }

  /** A platform oscillating along its own axis. `gap` = distance to the previous edge at the NEAR extreme. */
  mover(o: MoverOpts): this {
    const axisVec = (yawNow: number): Vec3 => {
      if (o.axis === 'up') return [0, 1, 0];
      if (o.axis === 'forward') return [-Math.sin(yawNow), 0, -Math.cos(yawNow)];
      return [Math.cos(yawNow), 0, -Math.sin(yawNow)];
    };
    // Base position: forward movers sit `amp` further ahead so their near extreme is `gap` away.
    const shift = o.axis === 'forward' ? o.amp : 0;
    const p = this.place('moving', o, undefined, shift);
    p.motion = { axis: axisVec(p.yaw), amp: o.amp, period: o.period, phase: o.phase ?? 0 } as MotionDef;
    if (o.axis === 'forward') {
      // advance the cursor to the FAR extreme front edge so the next platform keeps its gap
      const f = this.fwd();
      this.x += f[0] * o.amp;
      this.z += f[1] * o.amp;
    }
    return this;
  }

  /** Spinner bar standing on the most recent platform. */
  spinner(o: { length: number; speed: number; phase?: number; height?: number; thick?: number }): this {
    const p = this.platforms[this.platforms.length - 1];
    const h = o.height ?? 0.55;
    this.hazards.push({
      kind: 'spinner',
      pos: [p.pos[0], p.pos[1] + h / 2 + 0.02, p.pos[2]],
      size: [o.length, h, o.thick ?? 0.4],
      yaw: 0,
      spin: { speed: o.speed, phase: o.phase ?? 0 },
    });
    return this;
  }

  /** Wall sweeping sideways across the most recent platform. */
  slider(o: { width: number; height?: number; thick?: number; amp: number; period: number; phase?: number; at?: number }): this {
    const p = this.platforms[this.platforms.length - 1];
    const h = o.height ?? 1.3;
    const f = [-Math.sin(p.yaw), -Math.cos(p.yaw)];
    const r = [Math.cos(p.yaw), -Math.sin(p.yaw)];
    const along = (o.at ?? 0) * p.size[1] * 0.5;
    this.hazards.push({
      kind: 'slider',
      pos: [p.pos[0] + f[0] * along, p.pos[1] + h / 2 + 0.02, p.pos[2] + f[1] * along],
      size: [o.width, h, o.thick ?? 0.5],
      yaw: p.yaw,
      motion: { axis: [r[0], 0, r[1]], amp: o.amp, period: o.period, phase: o.phase ?? 0 },
    });
    return this;
  }

  build(): { platforms: PlatformDef[]; hazards: HazardDef[] } {
    return { platforms: this.platforms, hazards: this.hazards };
  }
}
