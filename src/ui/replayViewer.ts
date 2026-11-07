import * as THREE from 'three';
import type { ReplayRecord } from '../core/training/model';
import type { Viewport } from '../render/viewport';
import type { ReplayStore } from '../persist/replayStore';
import { STAGES } from '../core/tasks/obby/levels/compile';

const DT = 0.1; // seconds per recorded decision (6 physics steps at 60 Hz)
const EV_COLOR: Record<number, string> = { 1: '#94a3b8', 2: '#fbbf24', 3: '#ef4444', 4: '#34d399' };
const SPEEDS = [0.1, 0.25, 0.5, 1, 2, 4];

/** Training Replay: scrub/play recorded episodes, see checkpoints and falls, compare two episodes. */
export class ReplayViewer {
  private dock: HTMLElement;
  private a: ReplayRecord | null = null;
  private b: ReplayRecord | null = null;
  private idx = 0;
  private playing = true;
  private speed = 1;
  private raf = 0;
  private last = 0;
  private qBall = new THREE.Quaternion();
  private prev: THREE.Vector3 | null = null;

  constructor(private vp: Viewport, host: HTMLElement, private store: ReplayStore, private onClose: () => void) {
    this.dock = document.createElement('div');
    this.dock.className = 'replay-dock';
    this.dock.innerHTML = `
      <div class="row">
        <select class="sel" id="rp-a" aria-label="Episode"></select>
        <select class="sel" id="rp-b" aria-label="Compare with"></select>
        <button class="btn" id="rp-close">✕ Close</button>
      </div>
      <canvas id="rp-ticks"></canvas>
      <input type="range" id="rp-t" min="0" max="1" step="0.01" value="0" />
      <div class="row">
        <button class="btn primary" id="rp-play">❚❚ Pause</button>
        <button class="btn" id="rp-again">↻ Replay</button>
        <div class="seg" id="rp-speed">${SPEEDS.map((s) => `<button data-v="${s}">${s === 0.1 ? 'Slow-mo' : s + '×'}</button>`).join('')}</div>
        <span class="label-sm" id="rp-info" style="margin:0"></span>
      </div>`;
    host.appendChild(this.dock);
    this.vp.replayMode = true;
    const q = <T extends HTMLElement>(s: string) => this.dock.querySelector(s) as T;
    q('#rp-close').onclick = () => this.close();
    q('#rp-play').onclick = () => ((this.playing = !this.playing), this.syncUi());
    q('#rp-again').onclick = () => ((this.idx = 0), (this.playing = true), this.syncUi());
    q('#rp-speed').onclick = (e) => {
      const b = (e.target as HTMLElement).closest('button');
      if (b) (this.speed = Number(b.dataset.v)), this.syncUi();
    };
    q<HTMLSelectElement>('#rp-a').onchange = (e) => this.pickA(Number((e.target as HTMLSelectElement).value));
    q<HTMLSelectElement>('#rp-b').onchange = (e) => this.pickB(Number((e.target as HTMLSelectElement).value));
    q<HTMLInputElement>('#rp-t').oninput = (e) => {
      this.playing = false;
      this.idx = Number((e.target as HTMLInputElement).value);
      this.prev = null;
      this.syncUi();
    };
    this.fillA();
    const recs = this.store.records;
    if (recs.length) this.pickA(recs[recs.length - 1].ep);
    this.last = performance.now();
    const loop = (t: number) => {
      this.raf = requestAnimationFrame(loop);
      this.tick(Math.max(0, Math.min(0.1, (t - this.last) / 1000)));
      this.last = t;
    };
    this.raf = requestAnimationFrame(loop);
    this.syncUi();
  }

  private label(r: ReplayRecord): string {
    return `#${r.ep} · S${r.stage + 1} · ${Math.round(r.progress * 100)}% · ${r.success ? '✓ finished' : '✗'} · ${r.falls} falls · ${r.kind}`;
  }

  private fillA(): void {
    const sel = this.dock.querySelector('#rp-a') as HTMLSelectElement;
    const recs = this.store.records.slice().reverse();
    sel.innerHTML = recs.length ? recs.map((r) => `<option value="${r.ep}">${this.label(r)}</option>`).join('') : '<option>No episodes recorded yet</option>';
  }

  private pickA(ep: number): void {
    const r = this.store.byEp(ep);
    if (!r) return;
    this.a = r;
    this.b = null;
    (this.dock.querySelector('#rp-a') as HTMLSelectElement).value = String(ep);
    const sel = this.dock.querySelector('#rp-b') as HTMLSelectElement;
    const same = this.store.records.filter((x) => x.stage === r.stage && x.ep !== r.ep);
    sel.innerHTML = `<option value="0">Compare with…</option>` + same.map((x) => `<option value="${x.ep}">${x.ep < r.ep ? 'older ' : 'newer '}${this.label(x)}</option>`).join('');
    this.vp.setObbyStage(r.stage);
    this.idx = 0;
    this.prev = null;
    this.rebuildOverlays();
    this.syncUi();
  }

  private pickB(ep: number): void {
    this.b = ep ? (this.store.byEp(ep) ?? null) : null;
    this.rebuildOverlays();
  }

  private rebuildOverlays(): void {
    const a = this.a;
    this.vp.clearOverlays();
    if (!a) return;
    const n = a.trace.pos.length / 3;
    const ctx = (this.dock.querySelector('#rp-t') as HTMLInputElement);
    ctx.max = String(Math.max(1, this.len() - 1));
    if (this.b) this.vp.addTrail(this.b.trace.pos, '#fbbf24', 0.8);
    this.vp.addTrail(a.trace.pos, '#2dd4bf', 0.95);
    for (const [d, kind] of a.trace.events.map((e) => [e[0], e[1]] as const)) {
      if (kind === 1 || d >= n) continue;
      const p = a.trace.pos;
      this.vp.addMarker([p[d * 3] / 100, p[d * 3 + 1] / 100, p[d * 3 + 2] / 100], EV_COLOR[kind], kind === 3 ? 0.28 : 0.22);
    }
    this.drawTicks();
  }

  private len(): number {
    const a = this.a ? this.a.trace.pos.length / 3 : 0;
    const b = this.b ? this.b.trace.pos.length / 3 : 0;
    return Math.max(a, b);
  }

  private drawTicks(): void {
    const c = this.dock.querySelector('#rp-ticks') as HTMLCanvasElement;
    const w = (c.width = c.clientWidth * 2);
    const h = (c.height = 28);
    const g = c.getContext('2d')!;
    g.clearRect(0, 0, w, h);
    if (!this.a) return;
    const n = Math.max(1, this.a.trace.pos.length / 3 - 1);
    for (const e of this.a.trace.events) {
      if (e[1] === 1) continue;
      g.fillStyle = EV_COLOR[e[1]];
      g.beginPath();
      g.arc(8 + (e[0] / n) * (w - 16), h / 2, 6, 0, 7);
      g.fill();
    }
  }

  private pos(r: ReplayRecord, i: number): [number, number, number, boolean] {
    const p = r.trace.pos;
    const n = p.length / 3;
    const k = Math.min(n - 1, Math.floor(i));
    const k2 = Math.min(n - 1, k + 1);
    const f = Math.min(1, i - k);
    const A = [p[k * 3] / 100, p[k * 3 + 1] / 100, p[k * 3 + 2] / 100];
    const B = [p[k2 * 3] / 100, p[k2 * 3 + 1] / 100, p[k2 * 3 + 2] / 100];
    const jump = Math.hypot(B[0] - A[0], B[1] - A[1], B[2] - A[2]) > 3.2; // respawn: don't streak
    const t = jump ? 0 : f;
    return [A[0] + (B[0] - A[0]) * t, A[1] + (B[1] - A[1]) * t, A[2] + (B[2] - A[2]) * t, jump];
  }

  private tick(dt: number): void {
    const a = this.a;
    if (!a) return;
    const n = this.len();
    this.idx = Math.max(0, this.idx);
    if (this.playing) {
      this.idx += (dt * this.speed) / DT;
      if (this.idx >= n - 1) {
        this.idx = n - 1;
        this.playing = false;
        this.syncUi();
      }
      (this.dock.querySelector('#rp-t') as HTMLInputElement).value = String(this.idx);
    }
    const [x, y, z, jump] = this.pos(a, Math.min(this.idx, a.trace.pos.length / 3 - 1));
    const p = new THREE.Vector3(x, y, z);
    if (this.prev && !jump) {
      // visual rolling only (the real physics rotation is not stored)
      const d = p.clone().sub(this.prev);
      d.y = 0;
      const dist = d.length();
      if (dist > 1e-5) {
        const axis = new THREE.Vector3(0, 1, 0).cross(d).normalize();
        this.qBall.premultiply(new THREE.Quaternion().setFromAxisAngle(axis, dist / 0.5)).normalize();
      }
    }
    this.prev = p;
    const ghost = this.b ? this.pos(this.b, Math.min(this.idx, this.b.trace.pos.length / 3 - 1)) : null;
    this.vp.setReplayFrame(a.trace.t0 + this.idx * DT, [x, y, z], ghost ? [ghost[0], ghost[1], ghost[2]] : null, this.qBall);
    const t = this.idx * DT;
    const cp = a.trace.events.filter((e) => e[0] <= this.idx && e[1] === 2).length;
    const falls = a.trace.events.filter((e) => e[0] <= this.idx && e[1] === 3).length;
    (this.dock.querySelector('#rp-info') as HTMLElement).textContent =
      `t ${t.toFixed(1)}s · checkpoints ${cp}/${STAGES[a.stage].checkpointIdx.length} · falls ${falls}` +
      (this.b ? ` · amber = #${this.b.ep}` : '');
  }

  private syncUi(): void {
    (this.dock.querySelector('#rp-play') as HTMLElement).textContent = this.playing ? '❚❚ Pause' : '▶ Play';
    this.dock.querySelectorAll('#rp-speed button').forEach((b) => b.classList.toggle('on', Number((b as HTMLElement).dataset.v) === this.speed));
  }

  close(): void {
    cancelAnimationFrame(this.raf);
    this.dock.remove();
    this.vp.clearOverlays();
    this.vp.replayMode = false;
    this.onClose();
  }
}
