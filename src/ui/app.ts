import { SimClient } from '../app/simClient';
import { TASK_CATALOG } from '../core/tasks/catalog';
import type { TaskInfo } from '../core/task/types';
import { modelFromJson, modelToJson, ModelBlob, TrainingStats } from '../core/training/model';
import { deleteModel, deleteReplays, loadModel, loadReplays, loadSettings, saveModel, saveReplays, saveSettings, Settings } from '../persist/db';
import { ReplayStore } from '../persist/replayStore';
import { ObbyPanel } from './obbyPanel';
import { SkyPanel } from './skyPanel';
import { MusicPlayer } from '../audio/musicPlayer';
import { MusicPanel } from './musicPanel';
import { ReplayViewer } from './replayViewer';
import { BALL_PALETTES } from '../render/ballStyle';
import { QualityMode, Viewport } from '../render/viewport';
import type { FromWorker, RunStateName } from '../worker/protocol';
import { LearningGraphs } from './graphs';
import { confirmDialog, showTaskSelect, toast } from './taskSelect';

const SPEEDS: { label: string; value: number }[] = [
  { label: '1×', value: 1 },
  { label: '2×', value: 2 },
  { label: '4×', value: 4 },
  { label: '8×', value: 8 },
  { label: '16×', value: 16 },
  { label: 'MAX', value: 1_000_000 },
];
const QUALITIES: QualityMode[] = ['auto', 'high', 'medium', 'low'];
const AUTOSAVE_MS = 15000;

const STAT_DEFS: { key: string; label: string; hl?: boolean }[] = [
  { key: 'episode', label: 'Current Episode' },
  { key: 'trainSteps', label: 'Training Steps' },
  { key: 'totalScore', label: 'Total Score' },
  { key: 'successes', label: 'Successful Episodes' },
  { key: 'successRate', label: 'Success Rate', hl: true },
  { key: 'currentReward', label: 'Current Reward' },
  { key: 'avgReward', label: 'Average Reward' },
  { key: 'avgTime', label: 'Avg Completion Time' },
  { key: 'epsilon', label: 'Exploration Rate (ε)' },
];

export class App {
  private root: HTMLElement;
  viewport!: Viewport;
  private sim: SimClient | null = null;
  private graphs!: LearningGraphs;
  private task: TaskInfo | null = null;
  private runState: RunStateName = 'idle';
  private demo = false;
  private settings: Settings = { speed: 8, quality: 'auto', palette: 0 };
  private el: Record<string, HTMLElement> = {};
  private lastStats: TrainingStats | null = null;
  private lastUi = 0;
  private lastGraph = 0;
  private graphDirty = false;
  private lastSuccesses = -1;
  private saveTimer = 0;
  private fps = 0;
  private dps = 0;
  private tier = 'high';
  private obbyPanel: ObbyPanel | null = null;
  private skyPanel: SkyPanel | null = null;
  private replays = new ReplayStore();
  private replayViewer: ReplayViewer | null = null;
  private resumeAfterReplay = false;
  private isObby = false;

  constructor(root: HTMLElement) {
    this.root = root;
  }

  async boot(): Promise<void> {
    Object.assign(this.settings, await loadSettings());
    this.buildShell();
    this.viewport = new Viewport(this.el.viewport);
    this.skyPanel = new SkyPanel(this.viewport.sky, (source, values) => {
      this.settings.sky = { source, values };
      void saveSettings(this.settings);
    });
    this.el.side.appendChild(this.skyPanel.root);
    void this.skyPanel.init(this.settings.sky);
    const music = new MusicPlayer();
    music.configure({ ...this.settings.music });
    (window as unknown as { __music?: MusicPlayer }).__music = music;
    const mp = new MusicPanel(music, (m) => {
      this.settings.music = { ...m };
      void saveSettings(this.settings);
    });
    this.el.side.appendChild(mp.root);
    if (this.settings.music?.playing) {
      // browsers block autoplay: resume on the first tap/click
      const go = () => void music.play();
      window.addEventListener('pointerdown', go, { once: true });
    }
    (window as unknown as { __ballmind?: App }).__ballmind = this; // handy for debugging / automated tests
    this.viewport.setPalette(this.settings.palette);
    this.viewport.setQuality(this.settings.quality);
    this.viewport.onFps = (f, tier) => {
      this.fps = f;
      this.tier = tier;
      this.renderStatus();
    };
    this.viewport.start();
    // Draw an idle arena behind the task window straight away.
    this.viewport.buildTask(TASK_CATALOG[0].geometry);
    showTaskSelect(TASK_CATALOG, (t) => void this.startTask(t));
  }

  // ------------------------------------------------------------------ shell

  private buildShell(): void {
    this.root.innerHTML = `
    <div class="shell">
      <header class="topbar">
        <div class="brand"><i></i>BALLMIND</div>
        <button class="task-chip" id="taskChip" title="Change task">TASK ▾</button>
        <div class="spacer"></div>
        <div class="status"><span class="dot" id="dot"></span><span id="statusText">Idle</span><span class="fps" id="fps"></span></div>
      </header>

      <section class="viewport" id="viewport">
        <div class="hud">
          <div class="pill">Episode <b id="hudEp">1</b></div>
          <div class="pill">ε <b id="hudEps">1.00</b></div>
          <div class="pill">Action <b id="hudAct">—</b></div>
          <div class="pill demo" id="hudDemo" hidden>DEMO · greedy policy</div>
        </div>
        <div class="hint">Drag to orbit · Pinch to zoom</div>
        <div class="vp-tools"><button id="camReset" aria-label="Reset camera">⌖ Camera</button></div>
      </section>

      <aside class="side" id="side">
        <div id="obbyHost"></div>
        <section class="panel g-only">
          <h2>AI TRAINING</h2>
          <div class="stats" id="stats">
            ${STAT_DEFS.map(
              (s) =>
                `<div class="stat ${s.hl ? 'hl' : ''}"><label>${s.label}</label><span id="st-${s.key}">–</span></div>`,
            ).join('')}
          </div>
          <div class="label-sm" id="statsFoot"></div>
        </section>

        <section class="panel g-only">
          <h2>TRAINING CONTROLS</h2>
          <div class="row">
            <button class="btn primary" id="btnMain">▶ Start Training</button>
            <button class="btn" id="btnResetEnv">Reset Environment</button>
            <button class="btn danger" id="btnResetAI">Reset AI Training</button>
          </div>
          <div class="label-sm">Training Speed</div>
          <div class="seg" id="speedSeg" style="margin-top:6px">${SPEEDS.map(
            (s) => `<button data-v="${s.value}">${s.label}</button>`,
          ).join('')}</div>
          <div class="row" style="margin-top:10px">
            <button class="btn" id="btnDemo">Demo: watch best policy (ε = 0)</button>
          </div>
        </section>

        <section class="panel graphs g-only">
          <h2>LEARNING GRAPH</h2>
          <div class="gt"><span>Reward per episode <span style="color:#2dd4bf">— smoothed</span></span><span>episodes</span></div>
          <canvas id="gReward"></canvas>
          <div class="gt"><span>Success rate (rolling 50)</span><span>episodes</span></div>
          <canvas id="gSuccess"></canvas>
        </section>

        <section class="panel">
          <h2>MODEL &amp; DISPLAY</h2>
          <div class="row">
            <button class="btn" id="btnExport">Export Model</button>
            <button class="btn" id="btnImport">Import Model</button>
            <button class="btn" id="btnPalette">🎲 Ball Style</button>
          </div>
          <div class="label-sm">Graphics quality <span id="tierLbl"></span></div>
          <div class="seg" id="qualSeg" style="margin-top:6px">${QUALITIES.map(
            (q) => `<button data-v="${q}">${q[0].toUpperCase() + q.slice(1)}</button>`,
          ).join('')}</div>
          <div class="label-sm" id="saveInfo">Progress is saved automatically in this browser.</div>
          <input type="file" id="fileIn" accept=".json,application/json" hidden />
        </section>
      </aside>
    </div>`;

    const ids = [
      'viewport', 'taskChip', 'dot', 'statusText', 'fps', 'hudEp', 'hudEps', 'hudAct', 'hudDemo', 'camReset',
      'statsFoot', 'btnMain', 'btnResetEnv', 'btnResetAI', 'speedSeg', 'btnDemo', 'gReward', 'gSuccess',
      'obbyHost', 'side', 'btnExport', 'btnImport', 'btnPalette', 'qualSeg', 'saveInfo', 'fileIn', 'tierLbl',
    ];
    for (const id of ids) this.el[id] = this.root.querySelector('#' + id) as HTMLElement;
    for (const s of STAT_DEFS) this.el['st-' + s.key] = this.root.querySelector('#st-' + s.key) as HTMLElement;

    this.graphs = new LearningGraphs(this.el.gReward as HTMLCanvasElement, this.el.gSuccess as HTMLCanvasElement);

    this.obbyPanel = new ObbyPanel({
      send: (m) => this.sim?.send(m),
      toggleRun: () => this.toggleRun(),
      save: () => this.requestSave('manual'),
      resetAI: () => void this.resetAI(),
      openReplay: () => this.openReplay(),
      toggleDemo: () => this.toggleDemo(),
      setSpeed: (v) => this.setSpeed(v),
      setOverview: (on) => this.viewport.setOverview(on),
    });
    this.el.obbyHost.appendChild(this.obbyPanel.root);
    this.el.taskChip.addEventListener('click', () => this.changeTask());
    this.el.camReset.addEventListener('click', () => this.viewport.controls.reset());
    this.el.btnMain.addEventListener('click', () => this.toggleRun());
    this.el.btnResetEnv.addEventListener('click', () => this.sim?.send({ type: 'resetEnv' }));
    this.el.btnResetAI.addEventListener('click', () => void this.resetAI());
    this.el.btnDemo.addEventListener('click', () => this.toggleDemo());
    this.el.speedSeg.addEventListener('click', (e) => {
      const b = (e.target as HTMLElement).closest('button');
      if (!b) return;
      this.setSpeed(Number(b.dataset.v));
    });
    this.el.qualSeg.addEventListener('click', (e) => {
      const b = (e.target as HTMLElement).closest('button');
      if (!b) return;
      this.settings.quality = b.dataset.v as QualityMode;
      this.viewport.setQuality(this.settings.quality);
      void saveSettings(this.settings);
      this.renderControls();
    });
    this.el.btnPalette.addEventListener('click', () => {
      let next = this.viewport.palette;
      while (BALL_PALETTES.length > 1 && next === this.viewport.palette) next = Math.floor(Math.random() * BALL_PALETTES.length);
      this.viewport.setPalette(next);
      this.settings.palette = next;
      void saveSettings(this.settings);
      toast(`Ball style: ${BALL_PALETTES[next].name}`, false, 1200);
    });
    this.el.btnExport.addEventListener('click', () => this.sim?.send({ type: 'export' }));
    this.el.btnImport.addEventListener('click', () => (this.el.fileIn as HTMLInputElement).click());
    (this.el.fileIn as HTMLInputElement).addEventListener('change', (e) => void this.importFile(e));

    document.addEventListener('visibilitychange', () => {
      if (document.visibilityState === 'hidden') this.requestSave('hidden');
    });
    window.addEventListener('pagehide', () => this.requestSave('pagehide'));
    this.renderControls();
  }

  // ------------------------------------------------------------------ task / worker

  private async startTask(info: TaskInfo): Promise<void> {
    this.replayViewer?.close();
    this.task = info;
    this.isObby = info.id === 'obby-challenge';
    this.root.classList.toggle('is-obby', this.isObby);
    this.el.taskChip.textContent = info.title + ' ▾';
    if (this.isObby) this.viewport.buildObby();
    else this.viewport.buildTask(info.geometry);
    this.obbyPanel?.clear();
    this.replays.clear();
    if (this.isObby) this.replays.set(await loadReplays(info.id));
    this.sim?.terminate();
    clearInterval(this.saveTimer);
    this.graphs.clear();
    this.lastSuccesses = -1;
    this.demo = false;
    this.runState = 'idle';

    const saved = await loadModel(info.id);
    const sim = new SimClient();
    this.sim = sim;
    sim.onMessage = (m) => this.onWorker(m);
    sim.send({ type: 'init', taskId: info.id, seed: (Date.now() & 0xffff) + 1, model: saved });
    sim.send({ type: 'speed', value: this.settings.speed });
    this.saveTimer = window.setInterval(() => {
      if (this.runState === 'running' && !this.demo) this.requestSave('auto');
    }, AUTOSAVE_MS);
    this.renderControls();
    this.renderStatus();
  }

  private changeTask(): void {
    this.requestSave('switch');
    this.sim?.send({ type: 'pause' });
    showTaskSelect(TASK_CATALOG, (t) => void this.startTask(t));
  }

  private onWorker(m: FromWorker): void {
    switch (m.type) {
      case 'ready':
        this.graphs.setHistory(m.history);
        this.obbyPanel?.setHistory(m.history, m.loss);
        this.lastStats = m.stats;
        this.lastSuccesses = m.stats.successes;
        this.renderStats(true);
        if (m.restored) toast(`Restored AI: ${m.stats.episodesDone} episodes · ${m.stats.trainSteps} training steps`);
        break;
      case 'frame': {
        this.viewport.setSnapshot(m.snap);
        if (m.traces.length) this.replays.add(m.traces);
        for (const n of m.notices) toast(n.text, false, 4000);
        this.runState = m.state;
        this.dps = m.dps;
        this.lastStats = m.stats;
        if (m.episodes.length || m.loss.length) {
          this.graphs.push(m.episodes);
          this.obbyPanel?.push(m.episodes, m.loss);
          this.graphDirty = true;
        }
        if (!this.isObby && this.lastSuccesses >= 0 && m.stats.successes > this.lastSuccesses) this.viewport.flashTarget();
        this.lastSuccesses = m.stats.successes;
        if (this.task) this.el.hudAct.textContent = this.task.actionLabels[m.snap.action] ?? '—';
        const now = performance.now();
        if (now - this.lastUi > 120) {
          this.lastUi = now;
          this.renderStats();
          this.renderControls();
          this.renderStatus();
        }
        if (this.graphDirty && now - this.lastGraph > 400) {
          this.lastGraph = now;
          this.graphDirty = false;
          if (this.isObby) this.obbyPanel?.redrawGraphs();
          else this.graphs.redraw();
        }
        break;
      }
      case 'saved':
        if (m.reason === 'switch' || m.reason === 'hidden' || m.reason === 'pagehide' || !this.demo) {
          if (this.isObby) void saveReplays(m.model.taskId, this.replays.records);
          saveModel(m.model)
            .then(() => {
              if (m.reason === 'auto' || m.reason === 'manual')
                this.el.saveInfo.textContent = `Saved ${new Date().toLocaleTimeString()} · ${m.model.counters.episodes} episodes`;
            })
            .catch(() => (this.el.saveInfo.textContent = 'Could not save (storage unavailable)'));
        }
        break;
      case 'exported': {
        const blob = new Blob([modelToJson(m.model)], { type: 'application/json' });
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = `ballmind-${m.model.taskId}-ep${m.model.counters.episodes}.json`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        setTimeout(() => URL.revokeObjectURL(a.href), 4000);
        toast('Model exported');
        break;
      }
      case 'loaded':
        this.graphs.setHistory(m.history);
        this.obbyPanel?.setHistory(m.history, m.loss);
        this.lastStats = m.stats;
        this.lastSuccesses = m.stats.successes;
        this.renderStats(true);
        toast(`Model loaded: ${m.stats.episodesDone} episodes`);
        this.requestSave('manual');
        break;
      case 'reset-ai':
        this.graphs.clear();
        this.obbyPanel?.clear();
        this.replays.clear();
        this.lastStats = m.stats;
        this.lastSuccesses = 0;
        this.renderStats(true);
        break;
      case 'error':
        toast(m.message, true, 5000);
        break;
    }
  }

  private requestSave(reason: string): void {
    this.sim?.send({ type: 'save', reason });
  }

  // ------------------------------------------------------------------ actions

  private toggleDemo(): void {
    this.demo = !this.demo;
    this.sim?.send({ type: 'demo', on: this.demo });
    if (this.demo && this.runState === 'idle') this.sim?.send({ type: 'start' });
    this.renderControls();
  }

  private openReplay(): void {
    if (this.replayViewer || !this.isObby) return;
    this.resumeAfterReplay = this.runState === 'running';
    if (this.resumeAfterReplay) this.sim?.send({ type: 'pause' });
    this.replayViewer = new ReplayViewer(this.viewport, this.el.viewport, this.replays, () => {
      this.replayViewer = null;
      if (this.resumeAfterReplay) this.sim?.send({ type: 'resume' });
    });
  }

  private toggleRun(): void {
    if (!this.sim) return;
    if (this.demo) {
      // Leaving demo mode is explicit via the Demo button; the main button pauses/resumes.
    }
    if (this.runState === 'idle') this.sim.send({ type: 'start' });
    else if (this.runState === 'running') {
      this.sim.send({ type: 'pause' });
      this.requestSave('manual');
    } else this.sim.send({ type: 'resume' });
  }

  private setSpeed(v: number): void {
    this.settings.speed = v;
    this.sim?.send({ type: 'speed', value: v });
    void saveSettings(this.settings);
    this.renderControls();
  }

  private async resetAI(): Promise<void> {
    const ok = await confirmDialog(
      'RESET AI TRAINING',
      'This erases the neural network, replay memory, statistics and the saved copy in this browser. The AI starts again from zero.',
      'Reset AI',
    );
    if (!ok || !this.task) return;
    this.demo = false;
    this.sim?.send({ type: 'demo', on: false });
    this.sim?.send({ type: 'resetAI' });
    await deleteModel(this.task.id).catch(() => undefined);
    await deleteReplays(this.task.id);
    this.el.saveInfo.textContent = 'AI reset. Saved data cleared.';
    toast('AI training reset');
  }

  private async importFile(e: Event): Promise<void> {
    const input = e.target as HTMLInputElement;
    const f = input.files?.[0];
    input.value = '';
    if (!f) return;
    try {
      const model: ModelBlob = modelFromJson(await f.text());
      if (this.task && model.taskId !== this.task.id) throw new Error(`This model belongs to task "${model.taskId}"`);
      this.sim?.send({ type: 'load', model });
    } catch (err) {
      toast(`Import failed: ${(err as Error).message}`, true, 5000);
    }
  }

  // ------------------------------------------------------------------ render

  private renderControls(): void {
    const b = this.el.btnMain;
    const label =
      this.runState === 'idle' ? '▶ Start Training' : this.runState === 'running' ? '❚❚ Pause Training' : '▶ Resume Training';
    if (b.textContent !== label) b.textContent = label;
    b.classList.toggle('running', this.runState === 'running');
    this.el.btnDemo.classList.toggle('on', this.demo);
    this.el.hudDemo.hidden = !this.demo;
    this.el.speedSeg.querySelectorAll('button').forEach((x) => x.classList.toggle('on', Number((x as HTMLElement).dataset.v) === this.settings.speed));
    this.el.qualSeg.querySelectorAll('button').forEach((x) => x.classList.toggle('on', (x as HTMLElement).dataset.v === this.settings.quality));
    if (this.isObby && this.lastStats) this.obbyPanel?.update(this.lastStats, this.runState, this.demo, this.settings.speed);
    this.el.tierLbl.textContent = this.settings.quality === 'auto' ? `(auto → ${this.tier})` : '';
  }

  private renderStatus(): void {
    const names = { idle: 'Ready', running: this.demo ? 'Demo' : 'Training', paused: 'Paused' } as const;
    this.el.dot.className = 'dot ' + this.runState;
    this.el.statusText.textContent = names[this.runState];
    this.el.fps.textContent = this.fps ? ` · ${Math.round(this.fps)} fps` : '';
  }

  private renderStats(force = false): void {
    const s = this.lastStats;
    if (!s) return;
    const set = (k: string, v: string) => {
      const e = this.el['st-' + k];
      if (e.textContent !== v) e.textContent = v;
    };
    set('episode', String(s.episode));
    set('trainSteps', s.trainSteps.toLocaleString());
    set('totalScore', s.totalScore.toLocaleString());
    set('successes', s.successes.toLocaleString());
    set('successRate', `${Math.round(s.successRate * 100)}%`);
    set('currentReward', s.currentReward.toFixed(2));
    set('avgReward', s.avgReward.toFixed(2));
    set('avgTime', s.avgCompletionTime ? `${s.avgCompletionTime.toFixed(1)} s` : '–');
    set('epsilon', s.epsilon.toFixed(3));
    this.el.hudEp.textContent = String(s.episode);
    this.el.hudEps.textContent = s.epsilon.toFixed(2);
    const demoTxt = s.demo ? ` · demo ${s.demoEpisodes} eps, ${Math.round(s.demoSuccessRate * 100)}% success` : '';
    this.el.statsFoot.textContent =
      `Env steps ${s.envSteps.toLocaleString()} · replay ${s.bufferSize.toLocaleString()} · loss ${s.loss.toFixed(3)}` +
      (this.dps ? ` · ${Math.round(this.dps)} decisions/s` : '') +
      demoTxt;
    if (force) this.renderControls();
  }
}
