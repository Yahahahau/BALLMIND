export type MusicMode = 'loop' | 'shuffle';

export interface Track {
  title: string;
  artist: string;
  file: string;
}

export const TRACKS: Track[] = [
  { title: 'Pulse', artist: 'Veil44', file: 'music/pulse.mp3' },
  { title: 'bipolar', artist: 'diedlonely', file: 'music/bipolar.mp3' },
  { title: 'Two Years', artist: 'Victoriya', file: 'music/two-years.mp3' },
];

export interface MusicSettings {
  mode: MusicMode;
  volume: number;
  /** Was playing when the app was last closed: resumes on the first tap (browsers block autoplay). */
  playing: boolean;
}

export const DEFAULT_MUSIC: MusicSettings = { mode: 'loop', volume: 0.5, playing: false };

/**
 * Background music. Streams one track at a time (nothing is decoded up front), so it costs
 * almost nothing next to physics / training. "loop" plays 1→2→3→1…, "shuffle" picks a random
 * different track each time.
 */
export class MusicPlayer {
  readonly audio = new Audio();
  settings: MusicSettings = { ...DEFAULT_MUSIC };
  index = 0;
  onChange: () => void = () => undefined;
  /** Fired only when something worth saving changed (mode, volume, playing). */
  onSettings: () => void = () => undefined;
  private history: number[] = [];

  constructor() {
    this.audio.preload = 'none';
    this.audio.addEventListener('ended', () => this.next());
    this.audio.addEventListener('play', () => this.changed());
    this.audio.addEventListener('pause', () => this.changed());
    this.audio.addEventListener('timeupdate', () => this.onChange());
    this.audio.addEventListener('error', () => {
      // a missing/corrupt file must not stop the rest of the playlist
      if (this.audio.src) setTimeout(() => this.next(), 500);
    });
    this.load(0);
  }

  get track(): Track {
    return TRACKS[this.index];
  }
  get playing(): boolean {
    return !this.audio.paused;
  }

  configure(s: Partial<MusicSettings>): void {
    Object.assign(this.settings, s);
    this.audio.volume = Math.min(1, Math.max(0, this.settings.volume));
    this.onChange();
    this.onSettings();
  }

  private changed(): void {
    this.settings.playing = this.playing;
    this.onChange();
    this.onSettings();
  }

  private load(i: number): void {
    this.index = (i + TRACKS.length) % TRACKS.length;
    this.audio.src = new URL(TRACKS[this.index].file, document.baseURI).href;
  }

  async play(): Promise<void> {
    try {
      await this.audio.play();
    } catch {
      /* blocked until a user gesture: stay paused */
    }
  }
  pause(): void {
    this.audio.pause();
  }
  toggle(): void {
    if (this.playing) this.pause();
    else void this.play();
  }

  select(i: number): void {
    this.load(i);
    void this.play();
  }

  next(): void {
    this.history.push(this.index);
    if (this.history.length > 8) this.history.shift();
    let n: number;
    if (this.settings.mode === 'shuffle') {
      do n = Math.floor(Math.random() * TRACKS.length);
      while (n === this.index && TRACKS.length > 1);
    } else n = this.index + 1;
    this.load(n);
    void this.play();
  }

  prev(): void {
    if (this.audio.currentTime > 3) {
      this.audio.currentTime = 0;
      return;
    }
    const p = this.history.pop();
    this.load(p ?? this.index - 1);
    void this.play();
  }

  seek(frac: number): void {
    if (isFinite(this.audio.duration)) this.audio.currentTime = frac * this.audio.duration;
  }
}
