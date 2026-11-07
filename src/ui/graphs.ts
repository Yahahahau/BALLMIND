import type { EpisodeRecord } from '../core/training/model';

const MAX_POINTS = 260;

interface Series {
  values: number[];
  color: string;
  width: number;
  fill?: boolean;
}

/** Tiny dependency-free canvas line chart. */
class Chart {
  private ctx: CanvasRenderingContext2D;
  private w = 0;
  private h = 0;

  constructor(
    private canvas: HTMLCanvasElement,
    private fixedRange?: [number, number],
    private fmt: (v: number) => string = (v) => v.toFixed(1),
  ) {
    this.ctx = canvas.getContext('2d')!;
    new ResizeObserver(() => this.resize()).observe(canvas);
    this.resize();
  }

  private resize(): void {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = this.canvas.clientWidth;
    const h = this.canvas.clientHeight;
    if (!w || !h) return;
    this.w = w;
    this.h = h;
    this.canvas.width = Math.round(w * dpr);
    this.canvas.height = Math.round(h * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  draw(series: Series[], firstEp: number, lastEp: number): void {
    const { ctx, w, h } = this;
    if (!w) return this.resize();
    ctx.clearRect(0, 0, w, h);
    const padL = 36;
    const padR = 8;
    const padT = 8;
    const padB = 18;
    const pw = w - padL - padR;
    const ph = h - padT - padB;

    let lo = this.fixedRange?.[0] ?? Infinity;
    let hi = this.fixedRange?.[1] ?? -Infinity;
    if (!this.fixedRange) {
      for (const s of series) for (const v of s.values) (lo = Math.min(lo, v)), (hi = Math.max(hi, v));
      if (!isFinite(lo)) (lo = 0), (hi = 1);
      if (hi - lo < 1e-6) hi = lo + 1;
      const m = (hi - lo) * 0.08;
      lo -= m;
      hi += m;
    }
    const X = (i: number, n: number) => padL + (n <= 1 ? pw : (i / (n - 1)) * pw);
    const Y = (v: number) => padT + (1 - (v - lo) / (hi - lo)) * ph;

    ctx.font = '10px -apple-system, system-ui, sans-serif';
    ctx.fillStyle = '#8c99b0';
    ctx.strokeStyle = 'rgba(140,153,176,0.16)';
    ctx.lineWidth = 1;
    ctx.textAlign = 'right';
    for (let g = 0; g <= 3; g++) {
      const v = lo + ((hi - lo) * g) / 3;
      const y = Math.round(Y(v)) + 0.5;
      ctx.beginPath();
      ctx.moveTo(padL, y);
      ctx.lineTo(w - padR, y);
      ctx.stroke();
      ctx.fillText(this.fmt(v), padL - 5, y + 3);
    }
    ctx.textAlign = 'left';
    ctx.fillText(String(firstEp), padL, h - 4);
    ctx.textAlign = 'right';
    ctx.fillText(String(lastEp), w - padR, h - 4);

    for (const s of series) {
      const n = s.values.length;
      if (!n) continue;
      ctx.beginPath();
      s.values.forEach((v, i) => (i ? ctx.lineTo(X(i, n), Y(v)) : ctx.moveTo(X(i, n), Y(v))));
      if (s.fill) {
        const path = new Path2D();
        s.values.forEach((v, i) => (i ? path.lineTo(X(i, n), Y(v)) : path.moveTo(X(i, n), Y(v))));
        path.lineTo(X(n - 1, n), Y(lo));
        path.lineTo(X(0, n), Y(lo));
        path.closePath();
        ctx.fillStyle = s.color + '26';
        ctx.fill(path);
      }
      ctx.strokeStyle = s.color;
      ctx.lineWidth = s.width;
      ctx.lineJoin = 'round';
      ctx.stroke();
    }
  }
}

/** Learning graphs: reward per episode (raw + smoothed) and rolling success rate. */
export class LearningGraphs {
  private reward: Chart;
  private success: Chart;
  history: EpisodeRecord[] = [];

  constructor(rewardCanvas: HTMLCanvasElement, successCanvas: HTMLCanvasElement) {
    this.reward = new Chart(rewardCanvas);
    this.success = new Chart(successCanvas, [0, 1], (v) => `${Math.round(v * 100)}%`);
  }

  setHistory(h: EpisodeRecord[]): void {
    this.history = h.slice(-4000);
    this.redraw();
  }

  push(eps: EpisodeRecord[]): void {
    if (!eps.length) return;
    this.history.push(...eps);
    if (this.history.length > 4000) this.history.splice(0, this.history.length - 4000);
  }

  clear(): void {
    this.history = [];
    this.redraw();
  }

  redraw(): void {
    const h = this.history;
    const n = h.length;
    const first = n ? h[0].ep : 0;
    const last = n ? h[n - 1].ep : 0;
    const bucket = Math.max(1, Math.ceil(n / MAX_POINTS));
    const raw: number[] = [];
    const smooth: number[] = [];
    const rate: number[] = [];
    let ema = 0;
    let win = 0;
    const q: number[] = [];
    for (let i = 0; i < n; i++) {
      const r = h[i].reward;
      ema = i === 0 ? r : ema + (r - ema) * 0.08;
      q.push(h[i].success ? 1 : 0);
      win += q[q.length - 1];
      if (q.length > 50) win -= q[q.length - 51];
      if ((i + 1) % bucket === 0 || i === n - 1) {
        let s = 0;
        let c = 0;
        for (let j = Math.max(0, i - bucket + 1); j <= i; j++) (s += h[j].reward), c++;
        raw.push(s / c);
        smooth.push(ema);
        rate.push(win / Math.min(50, q.length));
      }
    }
    this.reward.draw(
      [
        { values: raw, color: '#5b6b85', width: 1 },
        { values: smooth, color: '#2dd4bf', width: 2.2 },
      ],
      first,
      last,
    );
    this.success.draw([{ values: rate, color: '#fbbf24', width: 2.2, fill: true }], first, last);
  }
}
