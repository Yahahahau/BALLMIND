import * as THREE from 'three';
import { HDRLoader } from 'three/examples/jsm/loaders/HDRLoader.js';
import { EXRLoader } from 'three/examples/jsm/loaders/EXRLoader.js';
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js';
import defaultSkyUrl from '../assets/sky-blue.jpg';

export type SkyKind = 'hdr' | 'exr' | 'equirect' | 'cube';
export type SkySource = 'default' | 'custom' | 'none';

/** Independent look controls. All are multipliers on top of the lighting adapted from the sky. */
export interface SkySettings {
  brightness: number; // background
  envIntensity: number; // image-based lighting on PBR materials
  exposure: number;
  contrast: number;
  saturation: number;
  ambient: number; // hemisphere light
  shadow: number; // 0..1
  adapt: boolean; // derive light colour/direction/strength from the sky
}

export const DEFAULT_SKY: SkySettings = {
  brightness: 1.15,
  envIntensity: 1,
  exposure: 1,
  contrast: 1.08,
  saturation: 1.1,
  ambient: 1,
  shadow: 1,
  adapt: true,
};

/** Raw files kept so the sky can be restored after reopening the app. */
export interface StoredSky {
  name: string;
  kind: SkyKind;
  files: { name: string; type: string; data: ArrayBuffer }[];
}

/** What the sky "looks like", used to colour the scene lighting. Colours are display-space (sRGB). */
export interface SkyAnalysis {
  zenith: THREE.Color;
  horizon: THREE.Color;
  ground: THREE.Color;
  sunDir: THREE.Vector3;
  sunColor: THREE.Color;
  /** 0 = overcast / no distinct sun, 1 = strong sun. */
  sunStrength: number;
  /** Mean brightness of the upper sky (0..1, tone-mapped). */
  luma: number;
}

export interface PreparedSky {
  name: string;
  kind: SkyKind;
  texture: THREE.Texture;
  analysis: SkyAnalysis;
  preview: HTMLCanvasElement;
  size: string;
  note: string;
  stored: StoredSky;
}

type Sampler = (u: number, vTop: number) => [number, number, number];

const MAX_SIDE = 4096;
const lin = (c: number) => (c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4));
const SRGB_LUT = Float32Array.from({ length: 256 }, (_, i) => lin(i / 255));
const toDisplay = (l: number) => Math.pow(Math.max(0, l / (1 + l)), 1 / 2.2);

export class SkyError extends Error {}

function ext(name: string): string {
  const m = /\.([a-z0-9]+)$/i.exec(name);
  return m ? m[1].toLowerCase() : '';
}

function sniff(buf: ArrayBuffer): 'hdr' | 'exr' | 'png' | 'jpg' | 'unknown' {
  const b = new Uint8Array(buf, 0, Math.min(16, buf.byteLength));
  const s = String.fromCharCode(...b.subarray(0, 10));
  if (s.startsWith('#?RADIANCE') || s.startsWith('#?RGBE')) return 'hdr';
  if (b[0] === 0x76 && b[1] === 0x2f && b[2] === 0x31 && b[3] === 0x01) return 'exr';
  if (b[0] === 0x89 && b[1] === 0x50 && b[2] === 0x4e && b[3] === 0x47) return 'png';
  if (b[0] === 0xff && b[1] === 0xd8 && b[2] === 0xff) return 'jpg';
  return 'unknown';
}

async function decodeImage(file: { data: ArrayBuffer; type: string; name: string }): Promise<{ img: HTMLImageElement; url: string }> {
  const url = URL.createObjectURL(new Blob([file.data], { type: file.type || 'image/png' }));
  const img = new Image();
  img.decoding = 'async';
  img.src = url;
  try {
    await img.decode();
  } catch {
    URL.revokeObjectURL(url);
    throw new SkyError(`"${file.name}" could not be decoded as an image`);
  }
  return { img, url };
}

const FACE_KEYS: Record<string, number> = {
  posx: 0, px: 0, right: 0, rt: 0, east: 0,
  negx: 1, nx: 1, left: 1, lf: 1, west: 1,
  posy: 2, py: 2, up: 2, top: 2,
  negy: 3, ny: 3, down: 3, dn: 3, bottom: 3, bot: 3,
  posz: 4, pz: 4, front: 4, ft: 4,
  negz: 5, nz: 5, back: 5, bk: 5,
};
const FACE_NAMES = ['+X', '−X', '+Y', '−Y', '+Z', '−Z'];

/** Order six files as +X, −X, +Y, −Y, +Z, −Z using face keywords in names; otherwise by file name. */
function orderFaces<T extends { name: string }>(files: T[]): { ordered: T[]; byName: boolean } {
  const slot: (T | undefined)[] = Array(6).fill(undefined);
  let ok = true;
  for (const f of files) {
    const tokens = f.name.replace(/\.[^.]+$/, '').toLowerCase().split(/[^a-z0-9]+/).filter(Boolean);
    const key = tokens.map((t) => FACE_KEYS[t]).find((v) => v !== undefined);
    if (key === undefined || slot[key]) {
      ok = false;
      break;
    }
    slot[key] = f;
  }
  if (ok) return { ordered: slot as T[], byName: true };
  return { ordered: files.slice().sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true })), byName: false };
}

// ------------------------------------------------------------------ samplers

function dataSampler(tex: THREE.DataTexture): Sampler {
  const { data, width: w, height: h } = tex.image as unknown as { data: Uint16Array | Float32Array; width: number; height: number };
  const half = data instanceof Uint16Array;
  const stride = Math.round(data.length / (w * h));
  const top = tex.flipY ? 0 : h - 1;
  const dir = tex.flipY ? 1 : -1;
  const rd = (i: number) => (half ? THREE.DataUtils.fromHalfFloat(data[i]) : data[i]);
  return (u, v) => {
    const x = Math.min(w - 1, Math.max(0, Math.floor(u * w)));
    const y = top + dir * Math.min(h - 1, Math.max(0, Math.floor(v * h)));
    const i = (y * w + x) * stride;
    return [Math.max(0, rd(i)), Math.max(0, rd(i + 1)), Math.max(0, rd(i + 2))];
  };
}

function canvasPixels(img: CanvasImageSource, w: number, h: number): Uint8ClampedArray {
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  const g = c.getContext('2d', { willReadFrequently: true })!;
  g.drawImage(img, 0, 0, w, h);
  return g.getImageData(0, 0, w, h).data;
}

function ldrSampler(img: HTMLImageElement): Sampler {
  const w = Math.min(512, img.naturalWidth);
  const h = Math.max(1, Math.round(w / 2));
  const px = canvasPixels(img, w, h);
  return (u, v) => {
    const x = Math.min(w - 1, Math.max(0, Math.floor(u * w)));
    const y = Math.min(h - 1, Math.max(0, Math.floor(v * h)));
    const i = (y * w + x) * 4;
    return [SRGB_LUT[px[i]], SRGB_LUT[px[i + 1]], SRGB_LUT[px[i + 2]]];
  };
}

/** Equirect view of a cubemap (faces: +X −X +Y −Y +Z −Z), using three.js' cube convention. */
function cubeSampler(faces: HTMLImageElement[]): Sampler {
  const n = Math.min(256, faces[0].naturalWidth);
  const px = faces.map((f) => canvasPixels(f, n, n));
  return (u, v) => {
    const phi = (u - 0.5) * Math.PI * 2;
    const th = (0.5 - v) * Math.PI;
    let x = Math.cos(th) * Math.cos(phi);
    const y = Math.sin(th);
    const z = Math.cos(th) * Math.sin(phi);
    x = -x; // CubeTexture is sampled mirrored in x
    const ax = Math.abs(x), ay = Math.abs(y), az = Math.abs(z);
    let f: number, s: number, t: number;
    if (ax >= ay && ax >= az) {
      f = x > 0 ? 0 : 1;
      s = x > 0 ? -z / ax : z / ax;
      t = -y / ax;
    } else if (ay >= az) {
      f = y > 0 ? 2 : 3;
      s = x / ay;
      t = y > 0 ? z / ay : -z / ay;
    } else {
      f = z > 0 ? 4 : 5;
      s = z > 0 ? x / az : -x / az;
      t = -y / az;
    }
    const X = Math.min(n - 1, Math.max(0, Math.floor(((s + 1) / 2) * n)));
    const Y = Math.min(n - 1, Math.max(0, Math.floor(((t + 1) / 2) * n)));
    const i = (Y * n + X) * 4;
    return [SRGB_LUT[px[f][i]], SRGB_LUT[px[f][i + 1]], SRGB_LUT[px[f][i + 2]]];
  };
}

// ------------------------------------------------------------------ analysis

function grid(s: Sampler, W: number, H: number): Float32Array {
  const out = new Float32Array(W * H * 3);
  for (let j = 0; j < H; j++)
    for (let i = 0; i < W; i++) {
      let r = 0, g = 0, b = 0;
      for (let dj = 0; dj < 2; dj++)
        for (let di = 0; di < 2; di++) {
          const c = s((i + (di + 0.5) / 2) / W, (j + (dj + 0.5) / 2) / H);
          r += c[0]; g += c[1]; b += c[2];
        }
      const o = (j * W + i) * 3;
      out[o] = r / 4; out[o + 1] = g / 4; out[o + 2] = b / 4;
    }
  return out;
}

export function analyse(s: Sampler): SkyAnalysis {
  const W = 96, H = 48;
  const g = grid(s, W, H);
  const acc = (j0: number, j1: number) => {
    let r = 0, gg = 0, b = 0, w = 0;
    for (let j = j0; j < j1; j++) {
      const wt = Math.cos((0.5 - (j + 0.5) / H) * Math.PI);
      for (let i = 0; i < W; i++) {
        const o = (j * W + i) * 3;
        r += g[o] * wt; gg += g[o + 1] * wt; b += g[o + 2] * wt; w += wt;
      }
    }
    return [r / w, gg / w, b / w];
  };
  const disp = (c: number[]) => new THREE.Color(toDisplay(c[0]), toDisplay(c[1]), toDisplay(c[2]));
  const lumaOf = (c: number[]) => 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
  const upper = acc(0, H * 0.3 | 0); // above ~+36°
  const mid = acc((H * 0.3) | 0, H / 2); // 0..36°
  const horizon = acc((H * 0.4) | 0, (H * 0.5) | 0);
  const ground = acc((H * 0.7) | 0, H);
  // brightest cell of the upper hemisphere = the sun (or the brightest part of an overcast sky)
  let best = -1, bi = 0, bj = 0, meanUp = 0, cnt = 0;
  for (let j = 0; j < H / 2; j++)
    for (let i = 0; i < W; i++) {
      const o = (j * W + i) * 3;
      const l = lumaOf([g[o], g[o + 1], g[o + 2]]);
      meanUp += l; cnt++;
      if (l > best) { best = l; bi = i; bj = j; }
    }
  meanUp /= cnt;
  const phi = ((bi + 0.5) / W - 0.5) * Math.PI * 2;
  const th = (0.5 - (bj + 0.5) / H) * Math.PI;
  const so = (bj * W + bi) * 3;
  const sc = [g[so], g[so + 1], g[so + 2]];
  const m = Math.max(sc[0], sc[1], sc[2], 1e-4);
  const sunColor = new THREE.Color(sc[0] / m, sc[1] / m, sc[2] / m);
  const contrast = best / (meanUp + 0.04);
  const sunStrength = Math.min(1, Math.max(0, (contrast - 1.6) / 5));
  return {
    zenith: disp(upper),
    horizon: disp(horizon),
    ground: disp(ground),
    sunDir: new THREE.Vector3(Math.cos(th) * Math.cos(phi), Math.sin(th), Math.cos(th) * Math.sin(phi)),
    sunColor,
    sunStrength,
    luma: toDisplay(lumaOf(upper) * 0.6 + lumaOf(mid) * 0.4),
  };
}

function previewCanvas(s: Sampler): HTMLCanvasElement {
  const W = 256, H = 128;
  const c = document.createElement('canvas');
  c.width = W;
  c.height = H;
  const g = c.getContext('2d')!;
  const img = g.createImageData(W, H);
  for (let j = 0; j < H; j++)
    for (let i = 0; i < W; i++) {
      const p = s((i + 0.5) / W, (j + 0.5) / H);
      const o = (j * W + i) * 4;
      img.data[o] = toDisplay(p[0]) * 255;
      img.data[o + 1] = toDisplay(p[1]) * 255;
      img.data[o + 2] = toDisplay(p[2]) * 255;
      img.data[o + 3] = 255;
    }
  g.putImageData(img, 0, 0);
  return c;
}

// ------------------------------------------------------------------ manager

interface Active {
  texture: THREE.Texture | null;
  env: THREE.WebGLRenderTarget | null;
  analysis: SkyAnalysis | null;
}

/**
 * Owns the scene background / environment map. Replaces and disposes previous textures and PMREM targets,
 * keeps a fallback environment, and never touches physics or AI.
 */
export class SkyboxManager {
  source: SkySource = 'default';
  name = 'Blue sky (default)';
  settings: SkySettings = { ...DEFAULT_SKY };
  analysis: SkyAnalysis | null = null;
  /** Called whenever the sky (and so the adapted lighting) changed. */
  onChange: () => void = () => undefined;

  private pmrem: THREE.PMREMGenerator;
  private fallbackEnv: THREE.WebGLRenderTarget;
  private active: Active = { texture: null, env: null, analysis: null };
  private readonly flat = new THREE.Color('#1b2130');

  constructor(private renderer: THREE.WebGLRenderer, private scene: THREE.Scene) {
    this.pmrem = new THREE.PMREMGenerator(renderer);
    this.fallbackEnv = this.pmrem.fromScene(new RoomEnvironment(), 0.04);
    this.useFlat();
  }

  // ---- loading / validation (nothing is applied here) ----

  async prepareFiles(files: File[]): Promise<PreparedSky> {
    if (!files.length) throw new SkyError('No file selected');
    const raw = await Promise.all(files.map(async (f) => ({ name: f.name, type: f.type, data: await f.arrayBuffer() })));
    return this.prepareStored(this.classify(raw));
  }

  private classify(raw: StoredSky['files']): StoredSky {
    if (raw.length === 6) return { name: raw[0].name.replace(/\.[^.]+$/, '').replace(/[-_ ]?(px|nx|py|ny|pz|nz|posx|negx|posy|negy|posz|negz|right|left|top|bottom|front|back|up|down)$/i, '') + ' (cubemap)', kind: 'cube', files: raw };
    if (raw.length !== 1) throw new SkyError(`Select either one .hdr / .exr / panorama image, or exactly six cubemap faces (you selected ${raw.length}).`);
    const f = raw[0];
    const real = sniff(f.data);
    const e = ext(f.name);
    if (real === 'unknown') throw new SkyError(`"${f.name}" is not a supported skybox file (expected .hdr, .exr, .png or .jpg).`);
    const kind: SkyKind = real === 'hdr' ? 'hdr' : real === 'exr' ? 'exr' : 'equirect';
    if ((e === 'hdr' && real !== 'hdr') || (e === 'exr' && real !== 'exr') || (['png', 'jpg', 'jpeg'].includes(e) && (real === 'hdr' || real === 'exr')))
      throw new SkyError(`"${f.name}": the extension says .${e} but the contents are ${real.toUpperCase()}.`);
    return { name: f.name, kind, files: raw };
  }

  /** Decode + validate stored data and build a texture. Throws SkyError with a readable message. */
  async prepareStored(st: StoredSky): Promise<PreparedSky> {
    const urls: string[] = [];
    try {
      if (st.kind === 'cube') {
        const { ordered, byName } = orderFaces(st.files);
        const dec = await Promise.all(ordered.map((f) => decodeImage(f)));
        dec.forEach((d) => urls.push(d.url));
        const w = dec[0].img.naturalWidth;
        dec.forEach((d, i) => {
          if (d.img.naturalWidth !== d.img.naturalHeight) throw new SkyError(`Cubemap face "${ordered[i].name}" is ${d.img.naturalWidth}×${d.img.naturalHeight}; faces must be square.`);
          if (d.img.naturalWidth !== w) throw new SkyError(`Cubemap faces must all be the same size ("${ordered[i].name}" is ${d.img.naturalWidth}px, expected ${w}px).`);
        });
        if (w > 2048) throw new SkyError(`Cubemap faces are ${w}px; use 2048px or smaller for iPad.`);
        const tex = await new THREE.CubeTextureLoader().loadAsync(urls);
        tex.colorSpace = THREE.SRGBColorSpace;
        const s = cubeSampler(dec.map((d) => d.img));
        return {
          name: st.name, kind: 'cube', texture: tex, analysis: analyse(s), preview: previewCanvas(s), size: `6 × ${w}×${w}`, stored: st,
          note: byName ? 'Faces matched by name (+X −X +Y −Y +Z −Z).' : `Faces ordered by file name as ${FACE_NAMES.join(', ')}. Rename them px nx py ny pz nz to be sure.`,
        };
      }
      const f = st.files[0];
      const url = URL.createObjectURL(new Blob([f.data], { type: f.type }));
      urls.push(url);
      let tex: THREE.Texture;
      let sampler: Sampler;
      let w: number, h: number;
      if (st.kind === 'equirect') {
        const { img } = await decodeImage(f);
        w = img.naturalWidth; h = img.naturalHeight;
        this.checkEquirect(f.name, w, h);
        tex = new THREE.Texture(img);
        tex.colorSpace = THREE.SRGBColorSpace;
        tex.needsUpdate = true;
        sampler = ldrSampler(img);
      } else {
        let dt: THREE.DataTexture;
        try {
          dt = st.kind === 'hdr' ? await new HDRLoader().loadAsync(url) : await new EXRLoader().loadAsync(url);
        } catch (e) {
          throw new SkyError(`"${f.name}" could not be read as ${st.kind.toUpperCase()}: ${(e as Error).message ?? e}`);
        }
        const img = dt.image as unknown as { width: number; height: number };
        w = img.width; h = img.height;
        this.checkEquirect(f.name, w, h);
        tex = dt;
        sampler = dataSampler(dt);
      }
      tex.mapping = THREE.EquirectangularReflectionMapping;
      tex.generateMipmaps = false;
      tex.minFilter = THREE.LinearFilter;
      tex.magFilter = THREE.LinearFilter;
      const ok = w === 2048 || w === 1024 || w === 4096 ? '' : ' (1024×512 or 2048×1024 recommended)';
      return {
        name: st.name, kind: st.kind, texture: tex, analysis: analyse(sampler), preview: previewCanvas(sampler), size: `${w}×${h}`, stored: st,
        note: (st.kind === 'equirect' ? 'Panorama image (LDR).' : st.kind === 'hdr' ? 'HDR (Radiance) — full dynamic range.' : 'OpenEXR — full dynamic range.') + ok,
      };
    } finally {
      // Image data is already decoded/copied; the blob URLs are no longer needed.
      setTimeout(() => urls.forEach((u) => URL.revokeObjectURL(u)), 0);
    }
  }

  private checkEquirect(name: string, w: number, h: number): void {
    if (Math.abs(w / h - 2) > 0.02) throw new SkyError(`"${name}" is ${w}×${h}. A 360° panorama must be 2:1 (e.g. 2048×1024).`);
    if (w > MAX_SIDE) throw new SkyError(`"${name}" is ${w}px wide. Use 1K–2K (max ${MAX_SIDE}px) so it runs smoothly on iPad.`);
    if (w < 256) throw new SkyError(`"${name}" is only ${w}px wide — too small for a skybox.`);
  }

  discard(p: PreparedSky): void {
    if (p.texture !== this.active.texture) p.texture.dispose();
  }

  // ---- applying ----

  apply(p: PreparedSky, source: SkySource = 'custom'): void {
    this.release();
    const env = p.kind === 'cube' ? this.pmrem.fromCubemap(p.texture as THREE.CubeTexture) : this.pmrem.fromEquirectangular(p.texture);
    this.active = { texture: p.texture, env, analysis: p.analysis };
    this.scene.background = p.texture;
    this.scene.environment = env.texture;
    this.source = source;
    this.name = p.name;
    this.analysis = p.analysis;
    this.applyIntensities();
    this.onChange();
  }

  /** Plain colour background + built-in studio environment (also the failure fallback). */
  remove(): void {
    this.release();
    this.useFlat();
    this.source = 'none';
    this.name = 'None (plain colour)';
    this.onChange();
  }

  async useDefault(): Promise<void> {
    const buf = await (await fetch(defaultSkyUrl)).arrayBuffer();
    const st: StoredSky = { name: 'Blue sky (default)', kind: 'equirect', files: [{ name: 'sky-blue.jpg', type: 'image/jpeg', data: buf }] };
    const p = await this.prepareStored(st);
    this.apply(p, 'default');
  }

  private useFlat(): void {
    this.scene.background = this.flat;
    this.scene.backgroundIntensity = 1;
    this.scene.environment = this.fallbackEnv.texture;
    this.scene.environmentIntensity = 0.3;
    this.analysis = null;
  }

  private release(): void {
    this.active.texture?.dispose();
    this.active.env?.dispose();
    this.active = { texture: null, env: null, analysis: null };
  }

  setSettings(s: Partial<SkySettings>): void {
    Object.assign(this.settings, s);
    this.applyIntensities();
    this.onChange();
  }

  applyIntensities(): void {
    if (!this.active.texture) return;
    this.scene.backgroundIntensity = this.settings.brightness;
    this.scene.environmentIntensity = 0.6 * this.settings.envIntensity;
  }

  /** Flat colour used for fog when no sky is applied. */
  get flatColor(): THREE.Color {
    return this.flat;
  }

  dispose(): void {
    this.release();
    this.fallbackEnv.dispose();
    this.pmrem.dispose();
  }
}
