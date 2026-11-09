import * as THREE from 'three';
import { SkyboxManager } from './skybox';
import { GtaoPipeline } from './gtao';
import type { EnvSnapshot } from '../core/task/types';
import { BallEyes } from './ballEyes';
import { visionCameraPose } from '../core/vision/camera';
import type { VisionSettings } from '../core/vision/config';
import { BALL_PALETTES, createCheckerTexture, createGridTexture } from './ballStyle';
import { CameraController } from './cameraController';
import { ObbyScene } from './obbyScene';
import { STAGES } from '../core/tasks/obby/levels/compile';

export type QualityMode = 'auto' | 'high' | 'medium' | 'low';
type Tier = 'high' | 'medium' | 'low';

const TIERS: Record<Tier, { dpr: number; shadow: number }> = {
  high: { dpr: 2, shadow: 2048 },
  medium: { dpr: 1.5, shadow: 1024 },
  low: { dpr: 1, shadow: 1024 },
};

/** The 3D view: clean minimal scene, soft real-time shadows, follow camera. */
export class Viewport {
  readonly renderer: THREE.WebGLRenderer;
  readonly scene = new THREE.Scene();
  readonly camera = new THREE.PerspectiveCamera(42, 1, 0.1, 120);
  readonly controls: CameraController;
  onFps: (fps: number, tier: Tier) => void = () => undefined;

  private readonly light: THREE.DirectionalLight;
  readonly sky: SkyboxManager;
  readonly gtao: GtaoPipeline;
  private hemi: THREE.HemisphereLight;
  private lightDir = new THREE.Vector3(7, 12, 5).normalize();
  private fogNear = 28;
  private fogFar = 62;
  private overviewOn = false;
  private ball!: THREE.Mesh;
  private eyes: BallEyes | null = null;
  private gazeVec: THREE.Vector3 | null = null;
  private gazeYaw = 0;
  private ballMat!: THREE.MeshPhysicalMaterial;
  private targetGroup = new THREE.Group();
  private targetMat!: THREE.MeshStandardMaterial;
  private world = new THREE.Group();
  private aniso: number;

  private snapPos = new THREE.Vector3(0, 0.5, 0);
  private snapQuat = new THREE.Quaternion();
  private snapTarget = new THREE.Vector3();
  private epoch = -1;
  private flash = 0;

  private mode: QualityMode = 'auto';
  private tier: Tier = 'high';
  private frameEma = 16;
  private slowChecks = 0;
  private lastCheck = 0;
  private fpsFrames = 0;
  private fpsStart = 0;
  private raf = 0;
  private lastT = 0;
  private resizeObs: ResizeObserver;
  private paletteIndex = 0;
  // ---- Obby ----
  private obby: ObbyScene | null = null;
  private obbyBall = false;
  /** Replay viewer drives the scene itself; live snapshots are ignored while set. */
  replayMode = false;
  private ghost: THREE.Mesh | null = null;
  private overlays = new THREE.Group();
  private overview = false;

  constructor(private readonly container: HTMLElement) {
    this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
    this.renderer.shadowMap.enabled = true;
    // PCFSoftShadowMap was folded into PCFShadowMap (soft, filtered) in recent three.js.
    this.renderer.shadowMap.type = THREE.PCFShadowMap;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 0.9;
    this.aniso = Math.min(8, this.renderer.capabilities.getMaxAnisotropy());
    container.appendChild(this.renderer.domElement);
    this.renderer.domElement.className = 'gl';

    // Sky + image-based light live in SkyboxManager (default blue sky, import, fallback environment).
    this.sky = new SkyboxManager(this.renderer, this.scene);
    this.gtao = new GtaoPipeline(this.renderer, this.scene, this.camera);
    this.scene.fog = new THREE.Fog(this.sky.flatColor, 28, 62);
    this.hemi = new THREE.HemisphereLight(0xdfe9ff, 0x3b3a40, 0.32);
    this.scene.add(this.hemi);

    // Key light. Shadow frustum is fitted tightly around the arena; bias values tuned visually.
    this.light = new THREE.DirectionalLight(0xfff1de, 2.0);
    this.light.position.set(7, 12, 5);
    this.light.castShadow = true;
    const sc = this.light.shadow.camera;
    sc.left = sc.bottom = -9.5;
    sc.right = sc.top = 9.5;
    sc.near = 2;
    sc.far = 32;
    this.light.shadow.bias = -0.0001;
    this.light.shadow.normalBias = 0.008;
    this.light.shadow.radius = 3;
    this.light.target.position.set(0, 0, 0);
    this.scene.add(this.light, this.light.target);

    this.scene.add(this.world);
    this.sky.onChange = () => this.applyAtmosphere();
    this.controls = new CameraController(this.camera, this.renderer.domElement);

    this.resizeObs = new ResizeObserver(() => this.resize());
    this.resizeObs.observe(container);
    this.applyTier('high');
    this.resize();
  }

  /** Build floor, target tile and ball for a task's geometry. */
  buildTask(g: Record<string, number>): void {
    this.resetWorld();
    this.obbyBall = false;
    this.controls.followScale = 0.65;
    this.controls.dist = 18;
    this.controls.setOverview(false, this.stageCenter, 30);
    const sc = this.light.shadow.camera;
    sc.left = sc.bottom = -9.5;
    sc.right = sc.top = 9.5;
    sc.far = 32;
    sc.updateProjectionMatrix();
    this.light.position.copy(this.lightDir).multiplyScalar(14.9);
    this.light.target.position.set(0, 0, 0);
    this.fogNear = 28;
    this.fogFar = 62;
    this.camera.far = 120;
    this.camera.updateProjectionMatrix();
    this.applyAtmosphere();
    const half = g.arenaHalf;
    const thick = g.floorThickness;

    const grid = createGridTexture(half * 2, this.aniso);
    const side = new THREE.MeshStandardMaterial({ color: '#c9c4ba', roughness: 0.9 });
    const top = new THREE.MeshStandardMaterial({ map: grid, roughness: 0.88, metalness: 0 });
    const floor = new THREE.Mesh(new THREE.BoxGeometry(half * 2, thick, half * 2), [
      side,
      side,
      top,
      side,
      side,
      side,
    ]);
    floor.position.y = -thick / 2;
    floor.receiveShadow = true;
    this.world.add(floor);

    // Target: bright tile on a darker base so it separates clearly from the floor.
    const th = g.targetHalf;
    this.targetMat = new THREE.MeshStandardMaterial({
      color: '#ff7a2f',
      emissive: '#ff5a00',
      emissiveIntensity: 0.12,
      roughness: 0.5,
    });
    const tile = new THREE.Mesh(new THREE.BoxGeometry(th * 2, 0.06, th * 2), this.targetMat);
    tile.position.y = 0.03;
    tile.receiveShadow = true;
    const rim = new THREE.Mesh(
      new THREE.BoxGeometry(th * 2 + 0.16, 0.03, th * 2 + 0.16),
      new THREE.MeshStandardMaterial({ color: '#3a2c24', roughness: 0.8 }),
    );
    rim.position.y = 0.015;
    rim.receiveShadow = true;
    this.targetGroup = new THREE.Group();
    this.targetGroup.add(rim, tile);
    this.world.add(this.targetGroup);

    this.ballMat = new THREE.MeshPhysicalMaterial({
      map: createCheckerTexture(BALL_PALETTES[this.paletteIndex], this.aniso),
      roughness: 0.34,
      metalness: 0.05,
      clearcoat: 0.5,
      clearcoatRoughness: 0.2,
    });
    this.ball = new THREE.Mesh(new THREE.SphereGeometry(g.ballRadius, 64, 48), this.ballMat);
    this.ball.castShadow = true;
    this.ball.receiveShadow = false;
    this.ball.position.set(g.startX, g.ballRadius, g.startZ);
    this.snapPos.copy(this.ball.position);
    this.world.add(this.ball);
    this.addEyes(g.ballRadius);
    this.epoch = -1;
  }

  /**
   * Derives lighting from the sky currently in the scene (colour, sun direction, brightness) and applies the
   * user's independent look controls. Does nothing to physics or the AI.
   */
  applyAtmosphere(): void {
    const sk = this.sky;
    const a = sk.analysis;
    const st = sk.settings;
    const base = { hemi: 0.32, key: 2.0, exposure: 0.9 };
    const tint = new THREE.Color();
    let keyI = base.key, hemiI = base.hemi;
    this.light.color.set(0xfff1de);
    this.hemi.color.set(0xdfe9ff);
    this.hemi.groundColor.set(0x3b3a40);
    this.lightDir.set(7, 12, 5).normalize();
    let fog = sk.flatColor;
    if (a && st.adapt) {
      // dark skies (storm / dusk) get less light, bright skies a bit more; colours follow the sky
      const f = Math.min(1.2, Math.max(0.7, 0.65 + 0.7 * a.luma));
      keyI *= f;
      hemiI *= 0.8 + 0.5 * a.luma;
      this.hemi.color.copy(tint.copy(a.zenith).lerp(new THREE.Color(1, 1, 1), 0.4));
      this.hemi.groundColor.copy(tint.copy(a.ground).multiplyScalar(0.6).lerp(new THREE.Color(0x3b3a40), 0.5));
      this.light.color.copy(new THREE.Color(0xfff1de).lerp(a.sunColor, 0.6 * a.sunStrength));
      if (a.sunStrength > 0.2) {
        const d = a.sunDir.clone();
        const el = Math.min(75, Math.max(38, (Math.asin(Math.max(-1, Math.min(1, d.y))) * 180) / Math.PI));
        const h = Math.hypot(d.x, d.z) || 1;
        const r = (el * Math.PI) / 180;
        this.lightDir.set((d.x / h) * Math.cos(r), Math.sin(r), (d.z / h) * Math.cos(r)).normalize();
      }
    }
    if (a) fog = tint.copy(a.horizon).multiplyScalar(0.92);
    this.light.intensity = keyI;
    this.hemi.intensity = hemiI * st.ambient;
    this.light.shadow.intensity = st.shadow;
    this.renderer.toneMappingExposure = base.exposure * st.exposure;
    this.renderer.domElement.style.filter = `contrast(${st.contrast}) saturate(${st.saturation})`;
    const fogO = this.scene.fog as THREE.Fog | null;
    if (fogO) {
      fogO.color.copy(fog);
      fogO.near = this.overviewOn ? 200 : this.fogNear;
      fogO.far = this.overviewOn ? 400 : this.fogFar;
    }
    if (!this.obbyBall) this.light.position.copy(this.lightDir).multiplyScalar(14.9);
  }

  private makeBall(radius: number): void {
    this.ballMat = new THREE.MeshPhysicalMaterial({
      map: createCheckerTexture(BALL_PALETTES[this.paletteIndex], this.aniso),
      roughness: 0.34,
      metalness: 0.05,
      clearcoat: 0.5,
      clearcoatRoughness: 0.2,
    });
    this.ball = new THREE.Mesh(new THREE.SphereGeometry(radius, 48, 36), this.ballMat);
    this.ball.castShadow = true;
    this.world.add(this.ball);
    this.addEyes(radius);
  }

  private addEyes(radius: number): void {
    this.eyes?.dispose();
    this.eyes = new BallEyes(radius);
    this.eyes.group.position.copy(this.ball.position);
    this.world.add(this.eyes.group);
  }

  /**
   * Gaze of the AI ball's eyes = the forward vector of the real AI vision camera (same pose function
   * the worker renders with). `null` for tasks without an AI camera: eyes look straight ahead.
   */
  setGaze(v: VisionSettings | null): void {
    if (!v) {
      this.gazeVec = null;
      return;
    }
    const f = visionCameraPose(v, 0, 0, 0).fwd;
    this.gazeVec = new THREE.Vector3(f[0], f[1], f[2]);
    // idle facing: the camera's heading when it looks ahead; when it looks (almost) straight down its heading
    // carries no meaning, so the face starts turned toward +Z (toward the default player view)
    this.gazeYaw = v.pitch >= 80 ? 0 : Math.atan2(Math.sin((v.yaw * Math.PI) / 180), -Math.cos((v.yaw * Math.PI) / 180));
  }

  private resetWorld(): void {
    this.world.traverse((o) => {
      const m = o as THREE.Mesh;
      if (m.geometry) m.geometry.dispose();
    });
    this.obby?.dispose();
    this.obby = null;
    this.world.clear();
    this.ghost = null;
    this.overlays = new THREE.Group();
    this.replayMode = false;
    this.overview = false;
    this.overviewOn = false;
  }

  /** Obby: floating platforms in the void. The ball is the same checkerboard ball. */
  buildObby(): void {
    this.resetWorld();
    this.obbyBall = true;
    this.obby = new ObbyScene();
    this.world.add(this.obby.group, this.overlays);
    this.makeBall(0.5);
    this.ball.position.set(0, 1, 0);
    this.snapPos.copy(this.ball.position);
    const sc = this.light.shadow.camera;
    sc.left = sc.bottom = -16;
    sc.right = sc.top = 16;
    sc.far = 70;
    sc.updateProjectionMatrix();
    this.fogNear = 40;
    this.fogFar = 90;
    this.camera.far = 160;
    this.camera.updateProjectionMatrix();
    this.applyAtmosphere();
    this.controls.dist = 13;
    this.controls.followScale = 1;
    this.setObbyStage(0);
    this.epoch = -1;
  }

  setObbyStage(i: number): void {
    if (!this.obby || this.obby.stageIdx === i) return;
    this.obby.setStage(i);
    const b = STAGES[i].bounds;
    this.stageCenter.set((b.min[0] + b.max[0]) / 2, (b.min[1] + b.max[1]) / 2, (b.min[2] + b.max[2]) / 2);
    this.stageSize = Math.max(b.max[0] - b.min[0], b.max[2] - b.min[2]);
    this.clearOverlays();
    this.epoch = -1;
  }
  private stageCenter = new THREE.Vector3();
  private stageSize = 30;

  setOverview(on: boolean): void {
    this.overview = on;
    this.controls.setOverview(on, this.stageCenter, this.stageSize);
    // The top-down camera sits far above the course: don't let fog swallow it.
    this.overviewOn = on;
    this.applyAtmosphere();
    this.camera.far = on ? 500 : 160;
    this.camera.updateProjectionMatrix();
  }

  // ---- replay helpers ----
  clearOverlays(): void {
    for (const c of [...this.overlays.children]) {
      this.overlays.remove(c);
      const m = c as THREE.Mesh | THREE.Line;
      m.geometry?.dispose();
      (m.material as THREE.Material)?.dispose?.();
    }
  }

  /** Trajectory line (cm int16 x,y,z) coloured per episode. */
  addTrail(pos: Int16Array, color: string, opacity = 0.9): void {
    const n = pos.length / 3;
    const a = new Float32Array(pos.length);
    for (let i = 0; i < pos.length; i++) a[i] = pos[i] / 100;
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(a, 3));
    const l = new THREE.Line(g, new THREE.LineBasicMaterial({ color, transparent: true, opacity }));
    l.frustumCulled = false;
    if (n > 1) this.overlays.add(l);
  }

  addMarker(p: [number, number, number], color: string, r = 0.22): void {
    const m = new THREE.Mesh(new THREE.SphereGeometry(r, 16, 12), new THREE.MeshBasicMaterial({ color }));
    m.position.set(p[0], p[1] + 0.9, p[2]);
    this.overlays.add(m);
  }

  /** Show the world at replay time `t` with the ball (and optional ghost ball) at the given positions. */
  setReplayFrame(t: number, ball: [number, number, number], ghost?: [number, number, number] | null, q?: THREE.Quaternion): void {
    if (![t, ...ball].every(Number.isFinite)) {
      console.warn('replay frame skipped (non-finite)', t, ball);
      return;
    }
    this.obby?.applyTime(t);
    this.snapPos.set(...ball);
    if (q) this.snapQuat.copy(q);
    if (!this.ghost) {
      this.ghost = new THREE.Mesh(
        new THREE.SphereGeometry(0.5, 24, 16),
        new THREE.MeshStandardMaterial({ color: '#fbbf24', transparent: true, opacity: 0.45, roughness: 0.4 }),
      );
      this.world.add(this.ghost);
    }
    this.ghost.visible = !!ghost;
    if (ghost) this.ghost.position.set(...ghost);
  }

  setPalette(i: number): void {
    this.paletteIndex = ((i % BALL_PALETTES.length) + BALL_PALETTES.length) % BALL_PALETTES.length;
    if (!this.ballMat) return;
    this.ballMat.map?.dispose();
    this.ballMat.map = createCheckerTexture(BALL_PALETTES[this.paletteIndex], this.aniso);
    this.ballMat.needsUpdate = true;
  }

  get palette(): number {
    return this.paletteIndex;
  }

  /** Latest physics state from the worker. */
  setSnapshot(s: EnvSnapshot): void {
    if (this.obbyBall) {
      if (this.replayMode || !this.obby) return;
      const b = s.bodies.ball;
      this.setObbyStage(s.info.stage ?? 0);
      this.snapPos.set(b.p[0], b.p[1], b.p[2]);
      this.snapQuat.set(b.q[0], b.q[1], b.q[2], b.q[3]);
      if (s.dyn) this.obby.applyDyn(s.dyn);
      if (s.epoch !== this.epoch) {
        this.epoch = s.epoch;
        this.ball.position.copy(this.snapPos);
        this.ball.quaternion.copy(this.snapQuat);
      }
      return;
    }
    const b = s.bodies.ball;
    const t = s.bodies.target;
    this.snapPos.set(b.p[0], b.p[1], b.p[2]);
    this.snapQuat.set(b.q[0], b.q[1], b.q[2], b.q[3]);
    this.snapTarget.set(t.p[0], t.p[1], t.p[2]);
    if (s.epoch !== this.epoch && this.ball) {
      this.epoch = s.epoch;
      this.ball.position.copy(this.snapPos);
      this.ball.quaternion.copy(this.snapQuat);
      this.targetGroup.position.copy(this.snapTarget);
    }
  }

  flashTarget(): void {
    this.flash = 1;
  }

  // -------- quality --------

  setQuality(mode: QualityMode): void {
    this.mode = mode;
    this.slowChecks = 0;
    this.applyTier(mode === 'auto' ? 'high' : mode);
  }

  private applyTier(t: Tier): void {
    this.tier = t;
    const cfg = TIERS[t];
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, cfg.dpr));
    const sh = this.light.shadow;
    if (sh.mapSize.x !== cfg.shadow) {
      sh.mapSize.set(cfg.shadow, cfg.shadow);
      sh.map?.dispose();
      sh.map = null;
    }
    this.resize();
  }

  private autoAdapt(now: number, dtMs: number): void {
    this.frameEma += (Math.min(dtMs, 100) - this.frameEma) * 0.05;
    if (this.mode !== 'auto' || now - this.lastCheck < 2000) return;
    this.lastCheck = now;
    // Sustained < ~40 fps => step the quality down (never back up, to avoid flip-flopping).
    this.slowChecks = this.frameEma > 25 ? this.slowChecks + 1 : 0;
    if (this.slowChecks >= 2) {
      this.slowChecks = 0;
      if (this.tier === 'high') this.applyTier('medium');
      else if (this.tier === 'medium') this.applyTier('low');
    }
  }

  // -------- loop --------

  resize(): void {
    const w = Math.max(1, this.container.clientWidth);
    const h = Math.max(1, this.container.clientHeight);
    this.renderer.setSize(w, h, false);
    this.renderer.domElement.style.width = '100%';
    this.renderer.domElement.style.height = '100%';
    this.camera.aspect = w / h;
    // Keep the whole arena in view in portrait layouts.
    this.camera.fov = w / h < 1 ? 56 : 42;
    this.camera.updateProjectionMatrix();
    this.gtao.resize();
  }

  start(): void {
    const loop = (t: number) => {
      this.raf = requestAnimationFrame(loop);
      const dtMs = this.lastT ? t - this.lastT : 16;
      this.lastT = t;
      this.frame(Math.min(0.05, dtMs / 1000), t, dtMs);
    };
    this.raf = requestAnimationFrame(loop);
  }

  private frame(dt: number, now: number, dtMs: number): void {
    if (this.ball) {
      const k = 1 - Math.exp(-38 * dt);
      if (!Number.isFinite(this.ball.position.x + this.ball.position.y + this.ball.position.z)) this.ball.position.copy(this.snapPos);
      this.ball.position.lerp(this.snapPos, k);
      this.ball.quaternion.slerp(this.snapQuat, k);
      this.eyes?.update(dt, this.ball.position, this.gazeVec, this.gazeYaw);
      if (this.obbyBall) {
        // shadow light follows the ball so a small, sharp shadow map covers the action
        const bp = this.ball.position;
        this.light.target.position.set(bp.x, bp.y, bp.z);
        this.light.position.set(bp.x + this.lightDir.x * 25, bp.y + this.lightDir.y * 25, bp.z + this.lightDir.z * 25);
        this.controls.update(dt, this.ball.position);
      } else {
        this.targetGroup.position.lerp(this.snapTarget, 1 - Math.exp(-30 * dt));
        this.controls.update(dt, this.ball.position);
      }
      if (!this.obbyBall && this.flash > 0) {
        this.flash = Math.max(0, this.flash - dt * 2.2);
        this.targetMat.emissiveIntensity = 0.12 + this.flash * 1.6;
      }
    }
    if (this.gtao.active) this.gtao.render();
    else this.renderer.render(this.scene, this.camera);

    this.autoAdapt(now, dtMs);
    this.gtao.adapt(now, this.frameEma);
    this.fpsFrames++;
    if (now - this.fpsStart >= 500) {
      this.onFps((this.fpsFrames * 1000) / (now - this.fpsStart), this.tier);
      this.fpsFrames = 0;
      this.fpsStart = now;
    }
  }

  dispose(): void {
    cancelAnimationFrame(this.raf);
    this.resizeObs.disconnect();
    this.renderer.dispose();
  }
}
