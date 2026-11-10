import { DEFAULT_VISION, sanitizeVision, VISION_LIMITS, type VisionSettings } from '../core/vision/config';
import type { FromWorker, ToWorker } from '../worker/protocol';

type Info = Extract<FromWorker, { type: 'visionInfo' }>;

/**
 * "AI VISION" block: what the AI camera is, its settings, and an optional live view of the EXACT image
 * the network receives. The feed is hidden (and not even streamed from the worker) unless toggled on.
 */
export class VisionPanel {
  readonly root = document.createElement('div');
  private q = <T extends HTMLElement>(s: string) => this.root.querySelector(s) as T;
  private draft: VisionSettings = { ...DEFAULT_VISION };
  private ctx: CanvasRenderingContext2D | null = null;
  private img: ImageData | null = null;

  constructor(
    private send: (m: ToWorker) => void,
    private persist: (s: VisionSettings) => void,
    private apply: (s: VisionSettings) => void,
  ) {
    this.root.className = 'vision-block';
    this.root.hidden = true;
    const L = VISION_LIMITS;
    this.root.innerHTML = `
      <div class="label-sm" style="margin-top:14px;color:var(--text);font-weight:700;letter-spacing:.06em">AI VISION · BALL CAMERA</div>
      <div class="label-sm" id="vi-info">—</div>
      <div class="sky-err" id="vi-note" hidden></div>
      <div class="label-sm">Pipeline: AI camera → RGB image → CNN encoder → visual features + physics sensors → Q-network → action</div>
      <div class="seg" id="vi-prev" style="margin-top:8px"><button data-v="0">Developer debug: OFF</button><button data-v="1">ON</button></div>
      <div id="vi-dbg" hidden>
      <div class="label-sm" id="vi-dtxt" style="white-space:pre-line;font-family:ui-monospace,monospace;margin-top:6px"></div>
      <canvas id="vi-cv" width="40" height="40" hidden style="width:160px;height:160px;image-rendering:pixelated;border:1px solid var(--line,#333);border-radius:8px;margin-top:8px"></canvas>
      </div>
      <div class="sliders" style="margin-top:10px">
        <label class="sl"><span>Image size <b id="vi-sv"></b></span><input type="range" id="vi-s" min="${L.size[0]}" max="${L.size[1]}" step="8"></label>
        <label class="sl top-only"><span>Frame stack <b id="vi-kv"></b></span><input type="range" id="vi-k" min="${L.stack[0]}" max="3" step="1"></label>
        <label class="sl"><span>Field of view <b id="vi-fv"></b></span><input type="range" id="vi-f" min="${L.fov[0]}" max="${L.fov[1]}" step="5"></label>
        <label class="sl top-only"><span>Camera pitch <b id="vi-pv"></b></span><input type="range" id="vi-p" min="${L.pitch[0]}" max="${L.pitch[1]}" step="5"></label>
        <label class="sl top-only"><span>Camera direction <b id="vi-yv"></b></span><input type="range" id="vi-y" min="${L.yaw[0]}" max="${L.yaw[1]}" step="15"></label>
        <label class="sl top-only"><span>Distance <b id="vi-dv"></b></span><input type="range" id="vi-d" min="${L.distance[0]}" max="${L.distance[1]}" step="1"></label>
        <label class="sl fp-only"><span>Gaze turn speed <b id="vi-yrv"></b></span><input type="range" id="vi-yr" min="${L.yawRate[0]}" max="${L.yawRate[1]}" step="15"></label>
        <label class="sl fp-only"><span>Gaze tilt speed <b id="vi-prv"></b></span><input type="range" id="vi-pr" min="${L.pitchRate[0]}" max="${L.pitchRate[1]}" step="10"></label>
      </div>
      <div class="row" style="margin-top:8px">
        <button class="btn" id="vi-apply">Apply camera settings</button>
        <button class="btn" id="vi-reset">Default</button>
      </div>
      <div class="label-sm" id="vi-bk"></div>
      <div class="label-sm">Changing image size or frame stack creates a new network (a different input shape). FOV / pitch / direction / distance only change what the AI sees; weights trained on another view may need to re-adapt.</div>`;
    this.q('#vi-prev').onclick = (e) => {
      const b = (e.target as HTMLElement).closest('button');
      if (b) this.preview(b.dataset.v === '1');
    };
    const bind = (id: string, key: keyof VisionSettings) => {
      this.q<HTMLInputElement>(id).oninput = (e) => {
        this.draft = sanitizeVision({ ...this.draft, [key]: Number((e.target as HTMLInputElement).value) });
        this.refresh();
      };
    };
    bind('#vi-s', 'size');
    bind('#vi-k', 'stack');
    bind('#vi-f', 'fov');
    bind('#vi-p', 'pitch');
    bind('#vi-y', 'yaw');
    bind('#vi-d', 'distance');
    bind('#vi-yr', 'yawRate');
    bind('#vi-pr', 'pitchRate');
    this.q('#vi-apply').onclick = () => {
      this.persist({ ...this.draft });
      this.apply({ ...this.draft });
    };
    this.q('#vi-reset').onclick = () => {
      this.draft = { ...DEFAULT_VISION };
      this.refresh();
    };
  }

  /** Called whenever a task loads. */
  setTask(isVision: boolean, saved?: VisionSettings): void {
    this.root.hidden = !isVision;
    if (saved) this.draft = sanitizeVision(saved);
    this.preview(false);
    this.refresh();
  }

  private preview(on: boolean): void {
    this.q('#vi-prev').querySelectorAll('button').forEach((b) => b.classList.toggle('on', (b.dataset.v === '1') === on));
    this.q('#vi-dbg').hidden = !on;
    this.q('#vi-cv').hidden = !on;
    this.send({ type: 'visionPreview', on });
  }

  onInfo(m: Info): void {
    this.draft = { ...m.settings };
    this.q('#vi-info').textContent = `${m.kind === 'gpu' ? 'WebGL2 GPU camera' : 'CPU camera (fallback)'} · ${m.settings.size}×${m.settings.size} RGB × ${m.settings.stack} frame${m.settings.stack > 1 ? 's' : ''} · CNN ${(m.params / 1000).toFixed(0)}k params · ${(m.macs / 1000).toFixed(0)}k MAC/forward`;
    const note = this.q('#vi-note');
    note.hidden = !m.note;
    note.textContent = m.note ?? '';
    this.refresh();
  }

  onFrame(size: number, rgb: Uint8Array): void {
    const cv = this.q<HTMLCanvasElement>('#vi-cv');
    if (cv.width !== size) {
      cv.width = size;
      cv.height = size;
      this.ctx = null;
    }
    this.ctx ??= cv.getContext('2d');
    if (!this.ctx) return;
    if (!this.img || this.img.width !== size) this.img = this.ctx.createImageData(size, size);
    const n = size * size;
    const d = this.img.data;
    for (let i = 0; i < n; i++) {
      d[i * 4] = rgb[i];
      d[i * 4 + 1] = rgb[n + i];
      d[i * 4 + 2] = rgb[2 * n + i];
      d[i * 4 + 3] = 255;
    }
    this.ctx.putImageData(this.img, 0, 0);
  }

  setBackupNote(t: string): void {
    this.q('#vi-bk').textContent = t;
  }

  /** Developer-debug readout (only meaningful while debug is on). */
  onStats(v: NonNullable<import('../core/training/model').TrainingStats['vision']>, i: { lag: number; rate: number; underruns: number }, inputOk: boolean): void {
    if (this.q('#vi-dbg').hidden) return;
    const fp = v.mode === 'fp';
    this.q('#vi-dtxt').textContent =
      `camera      ${fp ? 'first-person eye camera' : "bird's-eye (v1)"}  FOV ${v.fov}°\n` +
      (fp ? `yaw ${v.yaw.toFixed(0)}°  pitch ${v.pitch.toFixed(0)}° (limit ±80°, roll locked)\ngaze action ${v.gazeAction}\n` : '') +
      `vision rate ${v.decisionsPerSecond.toFixed(1)} frames/s (1 per decision)  render ${v.renderMs.toFixed(2)} ms\n` +
      `network input ${inputOk ? 'OK' : 'NO FRAMES'} · ${v.size}×${v.size}×3 · replay frames ${v.frames}\n` +
      `render sync  lag ${(i.lag * 1000).toFixed(0)} ms physics-time · sim ${i.rate.toFixed(2)}× · underruns ${i.underruns}`;
  }

  private refresh(): void {
    const s = this.draft;
    const set = (id: string, v: number, txt: string) => {
      const inp = this.q<HTMLInputElement>(id);
      if (document.activeElement !== inp) inp.value = String(v);
      this.q(id + 'v').textContent = txt;
    };
    set('#vi-s', s.size, `${s.size}×${s.size}`);
    set('#vi-k', s.stack, String(s.stack));
    set('#vi-f', s.fov, `${s.fov}°`);
    set('#vi-p', s.pitch, `${s.pitch}°`);
    set('#vi-y', s.yaw, `${s.yaw}°`);
    set('#vi-d', s.distance, `${s.distance} m`);
    set('#vi-yr', s.yawRate, `${s.yawRate}°/s`);
    set('#vi-pr', s.pitchRate, `${s.pitchRate}°/s`);
    set('#vi-f', s.fov, `${s.fov}°`);
    this.root.querySelectorAll<HTMLElement>('.top-only').forEach((e) => (e.hidden = s.mode === 'fp'));
    this.root.querySelectorAll<HTMLElement>('.fp-only').forEach((e) => (e.hidden = s.mode !== 'fp'));
  }
}
