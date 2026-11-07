import type { EpisodeRecord, StageStats, TrainingStats } from '../core/training/model';
import { STAGES } from '../core/tasks/obby/levels/compile';
import type { ToWorker, RunStateName } from '../worker/protocol';
import { Chart } from './graphs';

export interface ObbyApi {
  send(m: ToWorker): void;
  toggleRun(): void;
  save(): void;
  resetAI(): void;
  openReplay(): void;
  toggleDemo(): void;
  setSpeed(v: number): void;
  setOverview(on: boolean): void;
}

const SPEEDS = [
  { label: '1×', v: 1 },
  { label: '2×', v: 2 },
  { label: '4×', v: 4 },
  { label: 'Adaptive', v: 1_000_000 },
];
const DIFF_COLOR = ['#34d399', '#fbbf24', '#fb923c', '#f87171', '#c084fc'];

const STATS: [string, string][] = [
  ['stage', 'Current Stage'],
  ['episode', 'Current Episode'],
  ['episodes', 'Total Episodes'],
  ['trainSteps', 'Training Steps'],
  ['checkpoint', 'Current Checkpoint'],
  ['progress', 'Current Progress'],
  ['failures', 'Failure Count'],
  ['successes', 'Successful Attempts'],
  ['rate', 'Success Rate'],
  ['avgReward', 'Average Reward'],
  ['avgTime', 'Avg Completion Time'],
  ['bestTime', 'Best Completion Time'],
  ['epsilon', 'Epsilon (ε)'],
  ['loss', 'Training Loss'],
  ['model', 'Model Status'],
];

/** The Obby dashboard: stage selector, 15 live stats, controls, curriculum settings, 5 real graphs. */
export class ObbyPanel {
  readonly root: HTMLElement;
  private q = <T extends HTMLElement>(s: string) => this.root.querySelector(s) as T;
  private charts: Chart[] = [];
  private hist: EpisodeRecord[] = [];
  private loss: number[] = [];
  private viewStage = -1; // -1 = follow the current stage
  private stats: TrainingStats | null = null;
  private run: RunStateName = 'idle';
  private demo = false;
  private speed = 8;
  private overview = false;
  private lastKey = '';

  constructor(private api: ObbyApi) {
    this.root = document.createElement('div');
    this.root.className = 'obby-host';
    this.root.innerHTML = `
    <section class="panel">
      <h2>STAGES <span class="muted" id="ob-mode"></span></h2>
      <div class="chips" id="ob-chips"></div>
      <div class="label-sm" id="ob-blurb"></div>
    </section>
    <section class="panel">
      <h2>OBBY TRAINING</h2>
      <div class="stats">${STATS.map(([k, l]) => `<div class="stat ${k === 'rate' ? 'hl' : ''}"><label>${l}</label><span id="os-${k}">–</span></div>`).join('')}</div>
      <div class="label-sm" id="ob-foot"></div>
    </section>
    <section class="panel">
      <h2>CONTROLS</h2>
      <div class="row">
        <button class="btn primary" id="ob-main">▶ Start Training</button>
        <button class="btn" id="ob-stop">■ Stop</button>
        <button class="btn" id="ob-save">Save</button>
      </div>
      <div class="row" style="margin-top:8px">
        <button class="btn" id="ob-resetEnv">Reset Environment</button>
        <button class="btn danger" id="ob-resetAI">Reset AI Model</button>
      </div>
      <div class="label-sm">Training Speed</div>
      <div class="seg" id="ob-speed" style="margin-top:6px">${SPEEDS.map((s) => `<button data-v="${s.v}">${s.label}</button>`).join('')}</div>
      <div class="row" style="margin-top:10px">
        <button class="btn" id="ob-demo">Demo (ε = 0)</button>
        <button class="btn" id="ob-replay">▶ Training Replay</button>
        <button class="btn" id="ob-cam">Overview Camera</button>
      </div>
    </section>
    <section class="panel">
      <h2>CURRICULUM</h2>
      <div class="seg" id="ob-auto"><button data-v="1">Auto progression</button><button data-v="0">Manual stage</button></div>
      <div class="sliders">
        ${slider('threshold', 'Unlock success rate', 0.3, 1, 0.05, '%')}
        ${slider('minProgress', 'Min. average progress', 0.3, 1, 0.05, '%')}
        ${slider('window', 'Evaluation window (runs)', 5, 50, 1, '')}
        ${slider('timeFactor', 'Max time × par time', 1, 6, 0.5, '×')}
        ${slider('rehearsal', 'Rehearsal of earlier stages', 0, 0.5, 0.05, '%')}
      </div>
      <div class="label-sm">Start each run from (manual mode)</div>
      <div class="row" style="margin-top:6px">
        <div class="seg" id="ob-start"><button data-v="stage">Stage start</button><button data-v="checkpoint">Checkpoint</button></div>
        <select id="ob-cp" class="sel"></select>
      </div>
      <div class="label-sm" id="ob-cpnote">Checkpoint starts are practice only and never count as a full-stage success.</div>
    </section>
    <section class="panel graphs">
      <h2>LEARNING GRAPHS <span class="muted" id="ob-gview"></span></h2>
      <div class="gt"><span>Reward per episode</span><span></span></div><canvas id="og0"></canvas>
      <div class="gt"><span>Success rate (rolling 20)</span><span></span></div><canvas id="og1"></canvas>
      <div class="gt"><span>Completion time (s, successful runs)</span><span></span></div><canvas id="og2"></canvas>
      <div class="gt"><span>Checkpoint progress (best path %)</span><span></span></div><canvas id="og3"></canvas>
      <div class="gt"><span>Training loss</span><span>updates</span></div><canvas id="og4"></canvas>
    </section>`;
    this.charts = [
      new Chart(this.q('#og0')),
      new Chart(this.q('#og1'), [0, 1], (v) => `${Math.round(v * 100)}%`),
      new Chart(this.q('#og2'), undefined, (v) => v.toFixed(0)),
      new Chart(this.q('#og3'), [0, 1], (v) => `${Math.round(v * 100)}%`),
      new Chart(this.q('#og4'), undefined, (v) => v.toFixed(2)),
    ];
    this.wire();
  }

  private wire(): void {
    const cmd = (c: string, arg: unknown) => this.api.send({ type: 'command', cmd: c, arg });
    this.q('#ob-main').onclick = () => this.api.toggleRun();
    this.q('#ob-stop').onclick = () => this.api.send({ type: 'stop' });
    this.q('#ob-save').onclick = () => this.api.save();
    this.q('#ob-resetEnv').onclick = () => this.api.send({ type: 'resetEnv' });
    this.q('#ob-resetAI').onclick = () => this.api.resetAI();
    this.q('#ob-demo').onclick = () => this.api.toggleDemo();
    this.q('#ob-replay').onclick = () => this.api.openReplay();
    this.q('#ob-cam').onclick = () => {
      this.overview = !this.overview;
      this.api.setOverview(this.overview);
      this.q('#ob-cam').classList.toggle('on', this.overview);
    };
    this.q('#ob-speed').onclick = (e) => {
      const b = (e.target as HTMLElement).closest('button');
      if (b) this.api.setSpeed(Number(b.dataset.v));
    };
    this.q('#ob-auto').onclick = (e) => {
      const b = (e.target as HTMLElement).closest('button');
      if (!b) return;
      cmd('curriculum', { auto: b.dataset.v === '1' });
      this.api.send({ type: 'resetEnv' });
    };
    this.q('#ob-start').onclick = (e) => {
      const b = (e.target as HTMLElement).closest('button');
      if (!b) return;
      cmd('start', { startMode: b.dataset.v });
      this.api.send({ type: 'resetEnv' });
    };
    this.q<HTMLSelectElement>('#ob-cp').onchange = (e) => {
      cmd('start', { startCheckpoint: Number((e.target as HTMLSelectElement).value) });
      this.api.send({ type: 'resetEnv' });
    };
    this.q('#ob-chips').onclick = (e) => {
      const b = (e.target as HTMLElement).closest('button');
      if (!b) return;
      const i = Number(b.dataset.i);
      cmd('curriculum', { auto: false, manualStage: i });
      this.viewStage = i;
      this.api.send({ type: 'resetEnv' });
      this.redrawGraphs();
    };
    for (const inp of this.root.querySelectorAll<HTMLInputElement>('input[data-k]')) {
      inp.oninput = () => {
        const v = Number(inp.value);
        inp.parentElement!.querySelector('b')!.textContent = fmtSlider(inp.dataset.k!, v);
        cmd('curriculum', { [inp.dataset.k!]: v });
      };
    }
  }

  // ---------------------------------------------------------------- data in

  setHistory(h: EpisodeRecord[], loss: number[]): void {
    this.hist = h.slice(-4000);
    this.loss = loss.slice(-4000);
    this.redrawGraphs();
  }
  push(eps: EpisodeRecord[], loss: number[]): void {
    this.hist.push(...eps);
    if (this.hist.length > 4000) this.hist.splice(0, this.hist.length - 4000);
    this.loss.push(...loss);
    if (this.loss.length > 4000) this.loss.splice(0, this.loss.length - 4000);
  }
  clear(): void {
    this.hist = [];
    this.loss = [];
    this.redrawGraphs();
  }

  update(s: TrainingStats, run: RunStateName, demo: boolean, speed: number): void {
    this.stats = s;
    this.run = run;
    this.demo = demo;
    this.speed = speed;
    const x = s.extra as Record<string, any> | undefined;
    if (!x) return;
    const set = (k: string, v: string) => {
      const e = this.q('#os-' + k);
      if (e.textContent !== v) e.textContent = v;
    };
    const st = (x.stages as StageStats[] | undefined)?.[x.stage] ?? ({} as Partial<StageStats>);
    set('stage', `${x.stage + 1} · ${x.stageName}`);
    set('episode', String(s.episode));
    set('episodes', s.episodesDone.toLocaleString());
    set('trainSteps', s.trainSteps.toLocaleString());
    set('checkpoint', `${x.checkpoints} / ${x.checkpointTotal}`);
    set('progress', `${Math.round(x.progress * 100)}%`);
    set('failures', `${(st.falls ?? 0).toLocaleString()} <small>(run ${x.runFalls})</small>`.replace(/<[^>]+>/g, ''));
    set('successes', `${st.successes ?? 0} / ${st.episodes ?? 0}`);
    set('rate', st.attempts ? `${Math.round((st.successRate ?? 0) * 100)}%` : '–');
    set('avgReward', s.avgReward.toFixed(1));
    set('avgTime', st.avgTime ? `${st.avgTime.toFixed(1)} s` : '–');
    set('bestTime', st.bestTime ? `${st.bestTime.toFixed(1)} s` : '–');
    set('epsilon', s.epsilon.toFixed(3));
    set('loss', s.trainSteps ? s.loss.toFixed(3) : '–');
    set('model', this.modelStatus(s));
    this.q('#ob-foot').textContent =
      `Env steps ${s.envSteps.toLocaleString()} · replay ${s.bufferSize.toLocaleString()}` +
      (x.partial ? ' · PRACTICE RUN (checkpoint start, not counted)' : '') +
      (demo ? ` · demo ${s.demoEpisodes} runs, ${Math.round(s.demoSuccessRate * 100)}% success` : '');

    const key = JSON.stringify([x.unlocked, x.curriculum, x.startMode, x.startCheckpoint, x.stage, (x.stages as StageStats[]).map((a) => [a.episodes, Math.round(a.successRate * 20)]), x.unlockProgress.map((u: number) => Math.round(u * 20)), run, demo, speed]);
    if (key !== this.lastKey) {
      this.lastKey = key;
      this.renderStatic(x);
    }
  }

  private modelStatus(s: TrainingStats): string {
    if (this.demo) return 'Demo (greedy)';
    if (s.trainSteps === 0) return s.bufferSize > 0 ? 'Collecting data' : 'Untrained';
    if (this.run === 'running') return 'Learning';
    return this.run === 'paused' ? 'Paused' : 'Trained';
  }

  private renderStatic(x: Record<string, any>): void {
    const cur = x.curriculum as Record<string, number | boolean>;
    const stages = x.stages as StageStats[];
    this.q('#ob-mode').textContent = cur.auto ? '· auto progression' : '· manual';
    const chips = this.q('#ob-chips');
    chips.innerHTML = STAGES.map((s, i) => {
      const locked = cur.auto && i > x.unlocked;
      const rate = stages[i]?.attempts ? Math.round(stages[i].successRate * 100) + '%' : '–';
      const prog = i === x.unlocked && cur.auto ? Math.round(x.unlockProgress[i] * 100) : -1;
      return `<button data-i="${i}" class="chip ${i === x.stage ? 'on' : ''} ${locked ? 'locked' : ''}" style="--c:${DIFF_COLOR[i]}">
        <b>${i + 1}</b><span>${s.def.name}</span><small>${s.def.difficulty}${locked ? ' · 🔒' : ' · ' + rate}</small>
        ${prog >= 0 ? `<i style="width:${prog}%"></i>` : ''}</button>`;
    }).join('');
    this.q('#ob-blurb').textContent = STAGES[x.stage]?.def.blurb ?? '';
    this.q('#ob-auto').querySelectorAll('button').forEach((b) => b.classList.toggle('on', (b.dataset.v === '1') === !!cur.auto));
    this.q('#ob-start').querySelectorAll('button').forEach((b) => b.classList.toggle('on', b.dataset.v === x.startMode));
    const sel = this.q<HTMLSelectElement>('#ob-cp');
    const total = x.checkpointCounts[x.stage] as number;
    if (sel.options.length !== total) {
      sel.innerHTML = Array.from({ length: total }, (_, i) => `<option value="${i + 1}">Checkpoint ${i + 1}</option>`).join('');
    }
    sel.value = String(x.startCheckpoint);
    sel.disabled = !!cur.auto || x.startMode !== 'checkpoint';
    this.q('#ob-start').classList.toggle('disabled', !!cur.auto);
    for (const inp of this.root.querySelectorAll<HTMLInputElement>('input[data-k]')) {
      if (document.activeElement !== inp) inp.value = String(cur[inp.dataset.k!]);
      inp.parentElement!.querySelector('b')!.textContent = fmtSlider(inp.dataset.k!, Number(cur[inp.dataset.k!]));
    }
    const m = this.q('#ob-main');
    const label = this.run === 'idle' ? '▶ Start Training' : this.run === 'running' ? '❚❚ Pause' : '▶ Resume';
    if (m.textContent !== label) m.textContent = label;
    m.classList.toggle('running', this.run === 'running');
    this.q('#ob-demo').classList.toggle('on', this.demo);
    this.q('#ob-speed').querySelectorAll('button').forEach((b) => b.classList.toggle('on', Number(b.dataset.v) === this.speed));
  }

  // ---------------------------------------------------------------- graphs

  redrawGraphs(): void {
    const x = this.stats?.extra as Record<string, any> | undefined;
    const stage = this.viewStage >= 0 ? this.viewStage : (x?.stage ?? 0);
    this.q('#ob-gview').textContent = `· Stage ${stage + 1}`;
    const h = this.hist.filter((r) => r.stage === stage && !r.partial);
    const first = h[0]?.ep ?? 0;
    const last = h[h.length - 1]?.ep ?? 0;
    const bucket = (n: number) => Math.max(1, Math.ceil(n / 240));
    const down = (v: number[]) => {
      const b = bucket(v.length);
      if (b === 1) return v;
      const o: number[] = [];
      for (let i = 0; i < v.length; i += b) {
        const sl = v.slice(i, i + b);
        o.push(sl.reduce((a, c) => a + c, 0) / sl.length);
      }
      return o;
    };
    const ema = (v: number[]) => {
      let e = 0;
      return v.map((a, i) => (e = i ? e + (a - e) * 0.1 : a));
    };
    const rew = h.map((r) => r.reward);
    this.charts[0].draw([{ values: down(rew), color: '#5b6b85', width: 1 }, { values: down(ema(rew)), color: '#2dd4bf', width: 2.2 }], first, last);
    const rate: number[] = [];
    h.forEach((_, i) => {
      const w = h.slice(Math.max(0, i - 19), i + 1);
      rate.push(w.filter((r) => r.success).length / w.length);
    });
    this.charts[1].draw([{ values: down(rate), color: '#fbbf24', width: 2.2, fill: true }], first, last);
    const times = h.filter((r) => r.success).map((r) => r.steps * 0.1);
    this.charts[2].draw([{ values: down(times), color: '#38bdf8', width: 2 }], first, last);
    const prog = h.map((r) => r.progress ?? 0);
    this.charts[3].draw([{ values: down(prog), color: '#5b6b85', width: 1 }, { values: down(ema(prog)), color: '#34d399', width: 2.2, fill: true }], first, last);
    this.charts[4].draw([{ values: down(this.loss), color: '#f472b6', width: 2 }], 0, this.loss.length);
  }
}

function slider(k: string, label: string, min: number, max: number, step: number, _u: string): string {
  return `<label class="sl"><span>${label} <b></b></span><input type="range" data-k="${k}" min="${min}" max="${max}" step="${step}"></label>`;
}
function fmtSlider(k: string, v: number): string {
  if (k === 'threshold' || k === 'minProgress' || k === 'rehearsal') return `${Math.round(v * 100)}%`;
  if (k === 'timeFactor') return `${v}×`;
  return String(v);
}
