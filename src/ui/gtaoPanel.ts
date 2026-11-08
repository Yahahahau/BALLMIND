import { DEFAULT_GTAO, GtaoPipeline, GtaoQuality, GtaoSettings, PRESETS } from '../render/gtao';

const QUALS: GtaoQuality[] = ['low', 'medium', 'high', 'ultra'];

/** "GTAO · AMBIENT OCCLUSION" block inside the graphics settings. Every control drives the real pass. */
export class GtaoPanel {
  readonly root = document.createElement('div');
  private q = <T extends HTMLElement>(s: string) => this.root.querySelector(s) as T;

  constructor(private gtao: GtaoPipeline, private persist: (s: GtaoSettings) => void) {
    this.root.className = 'gtao-block';
    this.root.innerHTML = `
      <div class="label-sm" style="margin-top:14px;color:var(--text);font-weight:700;letter-spacing:.06em">GTAO · AMBIENT OCCLUSION</div>
      <div class="seg" id="ao-on" style="margin-top:6px"><button data-v="1">GTAO: ON</button><button data-v="0">OFF</button></div>
      <div class="label-sm">Quality <span id="ao-qinfo"></span></div>
      <div class="seg" id="ao-q" style="margin-top:6px">${QUALS.map((q) => `<button data-v="${q}">${q[0].toUpperCase() + q.slice(1)}</button>`).join('')}</div>
      <div class="sliders" style="margin-top:10px">
        <label class="sl"><span>Intensity <b id="ao-iv"></b></span><input type="range" id="ao-i" min="0" max="2" step="0.05"></label>
        <label class="sl"><span>Radius (m) <b id="ao-rv"></b></span><input type="range" id="ao-r" min="0.1" max="1.5" step="0.05"></label>
        <label class="sl"><span>Resolution scale <b id="ao-sv"></b></span><input type="range" id="ao-s" min="0.25" max="1" step="0.05"></label>
      </div>
      <div class="row" style="margin-top:8px">
        <button class="btn" id="ao-perf">Performance Mode</button>
        <button class="btn" id="ao-adapt">Adaptive resolution</button>
        <button class="btn" id="ao-reset">Reset to Default</button>
      </div>
      <div class="label-sm" id="ao-status"></div>
      <div class="sky-err" id="ao-err" hidden></div>`;
    this.q('#ao-on').onclick = (e) => {
      const b = (e.target as HTMLElement).closest('button');
      if (b) this.set({ enabled: b.dataset.v === '1' });
    };
    this.q('#ao-q').onclick = (e) => {
      const b = (e.target as HTMLElement).closest('button');
      if (b) this.set({ quality: b.dataset.v as GtaoQuality });
    };
    this.q<HTMLInputElement>('#ao-i').oninput = (e) => this.set({ intensity: Number((e.target as HTMLInputElement).value) });
    this.q<HTMLInputElement>('#ao-r').oninput = (e) => this.set({ radius: Number((e.target as HTMLInputElement).value) });
    this.q<HTMLInputElement>('#ao-s').oninput = (e) => this.set({ resScale: Number((e.target as HTMLInputElement).value) });
    this.q('#ao-perf').onclick = () => this.set({ performance: !gtao.settings.performance });
    this.q('#ao-adapt').onclick = () => this.set({ adaptive: !gtao.settings.adaptive });
    this.q('#ao-reset').onclick = () => this.set({ ...DEFAULT_GTAO });
    gtao.onChange = () => this.refresh();
    this.refresh();
  }

  private set(s: Partial<GtaoSettings>): void {
    this.gtao.apply(s);
    this.persist({ ...this.gtao.settings });
  }

  refresh(): void {
    const g = this.gtao;
    const s = g.settings;
    this.q('#ao-on').querySelectorAll('button').forEach((b) => b.classList.toggle('on', (b.dataset.v === '1') === s.enabled));
    this.q('#ao-q').querySelectorAll('button').forEach((b) => b.classList.toggle('on', b.dataset.v === s.quality));
    this.q('#ao-qinfo').textContent = `· ${PRESETS[s.quality].label}`;
    const set = (id: string, v: number, txt: string) => {
      const inp = this.q<HTMLInputElement>(id);
      if (document.activeElement !== inp) inp.value = String(v);
      this.q(id + 'v').textContent = txt;
    };
    set('#ao-i', s.intensity, s.intensity.toFixed(2));
    set('#ao-r', s.radius, `${s.radius.toFixed(2)} m`);
    set('#ao-s', s.resScale, `${Math.round(s.resScale * 100)}%`);
    this.q('#ao-perf').classList.toggle('on', s.performance);
    this.q('#ao-adapt').classList.toggle('on', s.adaptive);
    for (const id of ['#ao-q', '#ao-i', '#ao-r', '#ao-s', '#ao-perf', '#ao-adapt']) this.q(id).classList.toggle('disabled', !s.enabled);
    this.q('#ao-status').textContent = !s.enabled
      ? 'Off — standard rendering pipeline.'
      : `Active · ${g.effectiveSamples} samples · AO buffer ${Math.round(g.effectiveScale * 100)}%${g.autoFactor < 1 && s.adaptive ? ' (auto-reduced)' : ''}${s.performance ? ' · performance mode' : ''}`;
    const err = this.q('#ao-err');
    err.hidden = !g.message;
    err.textContent = g.message;
  }
}
