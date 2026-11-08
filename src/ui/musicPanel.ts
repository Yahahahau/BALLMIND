import { MusicMode, MusicPlayer, MusicSettings, TRACKS } from '../audio/musicPlayer';

const fmt = (s: number) => (isFinite(s) ? `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}` : '0:00');

/** MUSIC section: play/pause, previous/next, loop or shuffle across the three tracks, volume. */
export class MusicPanel {
  readonly root = document.createElement('section');
  private q = <T extends HTMLElement>(s: string) => this.root.querySelector(s) as T;

  constructor(private player: MusicPlayer, private persist: (s: MusicSettings) => void) {
    this.root.className = 'panel';
    this.root.innerHTML = `
      <h2>MUSIC</h2>
      <div class="sky-now"><b id="mu-title"></b><span class="label-sm" id="mu-time" style="margin:0 0 0 auto"></span></div>
      <input type="range" id="mu-seek" min="0" max="1000" value="0" style="width:100%;accent-color:var(--accent);margin-top:6px" aria-label="Seek" />
      <div class="row" style="margin-top:8px">
        <button class="btn" id="mu-prev" aria-label="Previous">⏮</button>
        <button class="btn primary" id="mu-play">▶ Play</button>
        <button class="btn" id="mu-next" aria-label="Next">⏭</button>
      </div>
      <div class="seg" id="mu-mode" style="margin-top:8px"><button data-v="loop">⟳ Loop all 3</button><button data-v="shuffle">⤮ Shuffle</button></div>
      <label class="sl" style="margin-top:8px"><span>Volume <b id="mu-volv"></b></span><input type="range" id="mu-vol" min="0" max="1" step="0.05"></label>
      <div class="seg" id="mu-list" style="margin-top:8px;flex-direction:column">${TRACKS.map((t, i) => `<button data-i="${i}">${t.title} — ${t.artist}</button>`).join('')}</div>`;
    this.q('#mu-play').onclick = () => player.toggle();
    this.q('#mu-next').onclick = () => player.next();
    this.q('#mu-prev').onclick = () => player.prev();
    this.q('#mu-mode').onclick = (e) => {
      const b = (e.target as HTMLElement).closest('button');
      if (!b) return;
      player.configure({ mode: b.dataset.v as MusicMode });
    };
    this.q('#mu-list').onclick = (e) => {
      const b = (e.target as HTMLElement).closest('button');
      if (b) player.select(Number(b.dataset.i));
    };
    this.q<HTMLInputElement>('#mu-vol').oninput = (e) => {
      player.configure({ volume: Number((e.target as HTMLInputElement).value) });
    };
    let seeking = false;
    const seek = this.q<HTMLInputElement>('#mu-seek');
    seek.onpointerdown = () => (seeking = true);
    seek.onchange = () => {
      player.seek(Number(seek.value) / 1000);
      seeking = false;
    };
    player.onChange = () => this.render(seeking);
    player.onSettings = () => persist(player.settings);
    this.render(false);
  }

  private render(seeking: boolean): void {
    const p = this.player;
    this.q('#mu-title').textContent = `${p.track.title} — ${p.track.artist}`;
    this.q('#mu-time').textContent = `${fmt(p.audio.currentTime)} / ${fmt(p.audio.duration)}`;
    this.q('#mu-play').textContent = p.playing ? '❚❚ Pause' : '▶ Play';
    if (!seeking && isFinite(p.audio.duration)) this.q<HTMLInputElement>('#mu-seek').value = String((p.audio.currentTime / p.audio.duration) * 1000);
    this.q('#mu-mode').querySelectorAll('button').forEach((b) => b.classList.toggle('on', b.dataset.v === p.settings.mode));
    this.q('#mu-list').querySelectorAll('button').forEach((b) => b.classList.toggle('on', Number(b.dataset.i) === p.index));
    this.q<HTMLInputElement>('#mu-vol').value = String(p.settings.volume);
    this.q('#mu-volv').textContent = `${Math.round(p.settings.volume * 100)}%`;
  }
}
