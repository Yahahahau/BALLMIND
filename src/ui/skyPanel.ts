import { DEFAULT_SKY, PreparedSky, SkyError, SkySettings } from '../render/skybox';
import type { SkyboxManager } from '../render/skybox';
import { deleteSky, loadSky, saveSky } from '../persist/db';

type Persist = (source: 'default' | 'custom' | 'none', values: SkySettings) => void;

const SLIDERS: { k: keyof SkySettings; label: string; min: number; max: number; step: number }[] = [
  { k: 'brightness', label: 'Skybox Brightness', min: 0.2, max: 2.5, step: 0.05 },
  { k: 'envIntensity', label: 'Environment Lighting', min: 0, max: 2.5, step: 0.05 },
  { k: 'exposure', label: 'Exposure', min: 0.5, max: 1.8, step: 0.05 },
  { k: 'contrast', label: 'Contrast', min: 0.8, max: 1.4, step: 0.02 },
  { k: 'saturation', label: 'Saturation', min: 0.6, max: 1.6, step: 0.02 },
  { k: 'ambient', label: 'Ambient Lighting', min: 0, max: 3, step: 0.05 },
  { k: 'shadow', label: 'Shadow Intensity', min: 0, max: 1, step: 0.05 },
];

/** "3D WORLD · SKYBOX" settings section: import, preview, apply, remove, restore default, look sliders. */
export class SkyPanel {
  readonly root = document.createElement('section');
  private pending: PreparedSky | null = null;
  private busy = false;
  private q = <T extends HTMLElement>(s: string) => this.root.querySelector(s) as T;

  constructor(private sky: SkyboxManager, private persist: Persist) {
    this.root.className = 'panel';
    this.root.innerHTML = `
      <h2>3D WORLD · SKYBOX</h2>
      <div class="sky-now"><span class="label-sm" style="margin:0">Current</span><b id="sk-name"></b><i class="spin" id="sk-spin" hidden></i></div>
      <div class="row" style="margin-top:8px">
        <button class="btn primary" id="sk-import">Import Skybox…</button>
        <button class="btn" id="sk-default">Restore Default</button>
        <button class="btn" id="sk-remove">Remove</button>
      </div>
      <input type="file" id="sk-file" multiple hidden />
      <div class="label-sm">HDRI <code>.hdr</code> / <code>.exr</code> (2:1), a 2:1 panorama <code>.png</code>/<code>.jpg</code>, or six cubemap faces (+X −X +Y −Y +Z −Z) selected together.</div>
      <div class="sky-err" id="sk-err" hidden></div>
      <div class="sky-prev" id="sk-prev" hidden>
        <canvas id="sk-canvas"></canvas>
        <div class="label-sm" id="sk-info"></div>
        <div class="row"><button class="btn primary" id="sk-apply">Apply</button><button class="btn" id="sk-cancel">Cancel</button></div>
      </div>
      <div class="sliders" style="margin-top:12px">
        ${SLIDERS.map((s) => `<label class="sl"><span>${s.label} <b></b></span><input type="range" data-k="${s.k}" min="${s.min}" max="${s.max}" step="${s.step}"></label>`).join('')}
      </div>
      <div class="row" style="margin-top:8px">
        <button class="btn" id="sk-adapt">Match lighting to sky</button>
        <button class="btn" id="sk-reset">Reset look</button>
      </div>
      <div class="label-sm">Sliders only change rendering — physics and AI are unaffected.</div>`;
    const file = this.q<HTMLInputElement>('#sk-file');
    this.q('#sk-import').onclick = () => file.click();
    file.onchange = () => {
      const f = [...(file.files ?? [])];
      file.value = '';
      if (f.length) void this.preview(f);
    };
    this.q('#sk-apply').onclick = () => this.apply();
    this.q('#sk-cancel').onclick = () => this.cancel();
    this.q('#sk-default').onclick = () => void this.restoreDefault();
    this.q('#sk-remove').onclick = () => {
      this.cancel();
      this.sky.remove();
      void deleteSky();
      this.save();
      this.refresh();
    };
    this.q('#sk-adapt').onclick = () => {
      this.sky.setSettings({ adapt: !this.sky.settings.adapt });
      this.save();
      this.refresh();
    };
    this.q('#sk-reset').onclick = () => {
      this.sky.setSettings({ ...DEFAULT_SKY });
      this.save();
      this.refresh();
    };
    for (const inp of this.root.querySelectorAll<HTMLInputElement>('input[data-k]')) {
      inp.oninput = () => {
        this.sky.setSettings({ [inp.dataset.k!]: Number(inp.value) } as Partial<SkySettings>);
        inp.parentElement!.querySelector('b')!.textContent = Number(inp.value).toFixed(2);
        this.save();
      };
    }
    this.refresh();
  }

  private save(): void {
    this.persist(this.sky.source, { ...this.sky.settings });
  }

  private setBusy(on: boolean): void {
    this.busy = on;
    this.q('#sk-spin').hidden = !on;
    for (const id of ['#sk-import', '#sk-default', '#sk-remove', '#sk-apply']) this.q<HTMLButtonElement>(id).disabled = on;
  }

  private error(msg: string | null): void {
    const e = this.q('#sk-err');
    e.hidden = !msg;
    e.textContent = msg ?? '';
  }

  refresh(): void {
    this.q('#sk-name').textContent = this.sky.name;
    this.q('#sk-adapt').classList.toggle('on', this.sky.settings.adapt);
    for (const inp of this.root.querySelectorAll<HTMLInputElement>('input[data-k]')) {
      const v = Number(this.sky.settings[inp.dataset.k as keyof SkySettings]);
      inp.value = String(v);
      inp.parentElement!.querySelector('b')!.textContent = v.toFixed(2);
    }
  }

  private async preview(files: File[]): Promise<void> {
    if (this.busy) return;
    this.cancel();
    this.error(null);
    this.setBusy(true);
    try {
      const p = await this.sky.prepareFiles(files);
      this.pending = p;
      const c = this.q<HTMLCanvasElement>('#sk-canvas');
      c.width = p.preview.width;
      c.height = p.preview.height;
      c.getContext('2d')!.drawImage(p.preview, 0, 0);
      this.q('#sk-info').textContent = `${p.name} · ${p.size} · ${p.note}`;
      this.q('#sk-prev').hidden = false;
    } catch (e) {
      this.error(e instanceof SkyError ? e.message : `Could not load skybox: ${(e as Error).message ?? e}`);
    } finally {
      this.setBusy(false);
    }
  }

  private apply(): void {
    const p = this.pending;
    if (!p) return;
    this.setBusy(true);
    try {
      this.sky.apply(p, 'custom');
      void saveSky(p.stored);
      this.pending = null;
      this.q('#sk-prev').hidden = true;
      this.save();
    } catch (e) {
      // keep whatever sky was showing; fall back to plain environment if even that is gone
      this.error(`Could not apply skybox: ${(e as Error).message ?? e}`);
      this.sky.discard(p);
    } finally {
      this.setBusy(false);
      this.refresh();
    }
  }

  private cancel(): void {
    if (this.pending) this.sky.discard(this.pending);
    this.pending = null;
    this.q('#sk-prev').hidden = true;
  }

  async restoreDefault(): Promise<void> {
    if (this.busy) return;
    this.cancel();
    this.error(null);
    this.setBusy(true);
    try {
      await this.sky.useDefault();
      await deleteSky();
      this.save();
    } catch (e) {
      this.sky.remove();
      this.error(`Default sky failed to load (${(e as Error).message ?? e}); using the plain background.`);
    } finally {
      this.setBusy(false);
      this.refresh();
    }
  }

  /** Startup: restore the remembered sky (or the default). Never throws. */
  async init(saved?: { source: 'default' | 'custom' | 'none'; values: SkySettings }): Promise<void> {
    if (saved?.values) this.sky.settings = { ...DEFAULT_SKY, ...saved.values };
    try {
      if (saved?.source === 'none') this.sky.remove();
      else if (saved?.source === 'custom') {
        const st = await loadSky();
        if (!st) throw new SkyError('saved skybox not found');
        this.sky.apply(await this.sky.prepareStored(st), 'custom');
      } else await this.sky.useDefault();
    } catch (e) {
      try {
        await this.sky.useDefault();
      } catch {
        this.sky.remove();
      }
      this.error(`Saved skybox could not be restored (${(e as Error).message ?? e}); using the default sky.`);
    }
    this.refresh();
  }
}
