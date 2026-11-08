import * as THREE from 'three';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { GTAOPass } from 'three/examples/jsm/postprocessing/GTAOPass.js';
import { OutputPass } from 'three/examples/jsm/postprocessing/OutputPass.js';

export type GtaoQuality = 'low' | 'medium' | 'high' | 'ultra';

export interface GtaoSettings {
  enabled: boolean;
  quality: GtaoQuality;
  /** Strength of the darkening (AO blend), 0..2. */
  intensity: number;
  /** World-space sampling radius in metres. */
  radius: number;
  /** AO buffer size relative to the screen (0.25..1). The preset sets it; you can fine-tune it. */
  resScale: number;
  /** Fewer samples, half-resolution cap, no MSAA — for low-powered devices. */
  performance: boolean;
  /** Lower the AO resolution automatically when the frame rate drops. */
  adaptive: boolean;
}

export const DEFAULT_GTAO: GtaoSettings = {
  enabled: true,
  quality: 'medium',
  intensity: 0.8,
  radius: 0.5,
  resScale: 0.75,
  performance: false,
  adaptive: true,
};

/** Every preset really changes the GTAO shader (sample count), the denoiser and the buffer size. */
export const PRESETS: Record<GtaoQuality, { samples: number; pdSamples: number; pdRings: number; resScale: number; msaa: number; label: string }> = {
  low: { samples: 8, pdSamples: 8, pdRings: 1, resScale: 0.5, msaa: 2, label: '8 samples · half-res' },
  medium: { samples: 12, pdSamples: 12, pdRings: 2, resScale: 0.75, msaa: 4, label: '12 samples · ¾-res' },
  high: { samples: 20, pdSamples: 16, pdRings: 2, resScale: 1, msaa: 4, label: '20 samples · full-res' },
  ultra: { samples: 32, pdSamples: 24, pdRings: 3, resScale: 1, msaa: 4, label: '32 samples · full-res' },
};

export type GtaoStatus = 'on' | 'off' | 'error';

/**
 * Real GTAO (three.js GTAOPass: depth/normal G-buffer → horizon-based AO → Poisson denoise → multiply).
 * Pipeline when ON:  RenderPass (MSAA, half-float) → GTAOPass → OutputPass (tone mapping + sRGB).
 * When OFF the composer is disposed and the original `renderer.render` path is used untouched.
 * Shadow maps are rendered by the normal scene pass and are not touched.
 */
export class GtaoPipeline {
  settings: GtaoSettings = { ...DEFAULT_GTAO };
  status: GtaoStatus = 'off';
  message = '';
  /** Automatic multiplier (0.5..1) applied to the resolution when the frame rate is low. */
  autoFactor = 1;
  onChange: () => void = () => undefined;

  private composer: EffectComposer | null = null;
  private pass: GTAOPass | null = null;
  private builtKey = '';
  private lastCheck = 0;
  private goodChecks = 0;
  private readonly size = new THREE.Vector2();

  constructor(private renderer: THREE.WebGLRenderer, private scene: THREE.Scene, private camera: THREE.PerspectiveCamera) {}

  get active(): boolean {
    return this.settings.enabled && this.status === 'on';
  }

  /** AO buffer scale actually used. */
  get effectiveScale(): number {
    const s = this.settings;
    let v = s.resScale * (s.adaptive ? this.autoFactor : 1);
    if (s.performance) v = Math.min(v, 0.5);
    return Math.min(1, Math.max(0.25, v));
  }

  get effectiveSamples(): number {
    const p = PRESETS[this.settings.quality];
    return this.settings.performance ? Math.max(6, p.samples >> 1) : p.samples;
  }

  /** Apply (partial) settings immediately. */
  apply(s: Partial<GtaoSettings>): void {
    const prevQ = this.settings.quality;
    Object.assign(this.settings, s);
    if (s.quality && s.quality !== prevQ && s.resScale === undefined) this.settings.resScale = PRESETS[s.quality].resScale;
    if (s.resScale !== undefined || s.quality) this.autoFactor = 1;
    this.sync();
    this.onChange();
  }

  /** (Re)build or tear down as needed, then push parameters into the shader. */
  sync(): void {
    if (!this.settings.enabled) {
      this.teardown();
      this.status = 'off';
      this.message = '';
      return;
    }
    if (this.status === 'error') this.status = 'off';
    const p = PRESETS[this.settings.quality];
    const msaa = this.settings.performance ? 0 : p.msaa;
    const key = `${msaa}`;
    try {
      if (!this.composer || this.builtKey !== key) this.build(msaa);
      this.status = 'on';
      this.message = '';
      this.configure();
    } catch (e) {
      this.fail(e);
    }
  }

  private build(msaa: number): void {
    this.teardown();
    const gl = this.renderer;
    if (!gl.capabilities.isWebGL2) throw new Error('WebGL2 is required for GTAO');
    const half = gl.extensions.has('EXT_color_buffer_half_float') || gl.extensions.has('EXT_color_buffer_float');
    gl.getDrawingBufferSize(this.size);
    const w = Math.max(2, this.size.x), h = Math.max(2, this.size.y);
    const rt = new THREE.WebGLRenderTarget(w, h, {
      type: half ? THREE.HalfFloatType : THREE.UnsignedByteType,
      samples: msaa,
    });
    const composer = new EffectComposer(gl, rt);
    composer.addPass(new RenderPass(this.scene, this.camera));
    const pass = new GTAOPass(this.scene, this.camera, w, h);
    pass.output = GTAOPass.OUTPUT.Default;
    composer.addPass(pass);
    composer.addPass(new OutputPass());
    this.composer = composer;
    this.pass = pass;
    this.builtKey = `${msaa}`;
    this.resize();
  }

  private configure(): void {
    const pass = this.pass;
    if (!pass) return;
    const p = PRESETS[this.settings.quality];
    const perf = this.settings.performance;
    pass.blendIntensity = this.settings.intensity;
    pass.updateGtaoMaterial({
      radius: this.settings.radius,
      distanceExponent: 1,
      thickness: 1,
      distanceFallOff: 1,
      scale: 1,
      samples: this.effectiveSamples,
      screenSpaceRadius: false,
    });
    pass.updatePdMaterial({
      lumaPhi: 10,
      depthPhi: 2,
      normalPhi: 3,
      radius: perf ? 4 : 6,
      radiusExponent: 1,
      rings: perf ? 1 : p.pdRings,
      samples: perf ? 8 : p.pdSamples,
    });
    this.resize();
  }

  /** Call after the renderer size / pixel ratio changed, or when the AO scale changed. */
  resize(): void {
    if (!this.composer || !this.pass) return;
    this.renderer.getDrawingBufferSize(this.size);
    const w = Math.max(2, this.size.x), h = Math.max(2, this.size.y);
    this.composer.setSize(w, h); // full-resolution scene colour
    const k = this.effectiveScale;
    this.pass.setSize(Math.max(2, Math.round(w * k)), Math.max(2, Math.round(h * k))); // AO + G-buffer at reduced size
  }

  render(): void {
    if (!this.composer) return;
    try {
      this.composer.render();
    } catch (e) {
      this.fail(e);
      this.renderer.setRenderTarget(null);
      this.renderer.render(this.scene, this.camera);
    }
  }

  /** Automatic resolution scaling: step down quickly when slow, creep back up when comfortably fast. */
  adapt(now: number, frameMs: number): void {
    if (!this.active || !this.settings.adaptive || now - this.lastCheck < 2000) return;
    this.lastCheck = now;
    if (frameMs > 24 && this.autoFactor > 0.5) {
      this.autoFactor = Math.max(0.5, this.autoFactor - 0.125);
      this.goodChecks = 0;
      this.resize();
      this.onChange();
    } else if (frameMs < 14 && this.autoFactor < 1) {
      if (++this.goodChecks >= 4) {
        this.autoFactor = Math.min(1, this.autoFactor + 0.125);
        this.goodChecks = 0;
        this.resize();
        this.onChange();
      }
    } else this.goodChecks = 0;
  }

  private fail(e: unknown): void {
    console.warn('GTAO disabled after an error:', e);
    this.teardown();
    this.status = 'error';
    this.message = `GTAO was turned off automatically (${(e as Error)?.message ?? e}). Rendering continues without it.`;
    this.settings.enabled = false;
    this.onChange();
  }

  private teardown(): void {
    if (this.composer) {
      this.pass?.dispose();
      this.composer.renderTarget1.dispose();
      this.composer.renderTarget2.dispose();
      this.composer.dispose();
    }
    this.composer = null;
    this.pass = null;
    this.builtKey = '';
    this.renderer.setRenderTarget(null);
  }

  dispose(): void {
    this.teardown();
  }
}
