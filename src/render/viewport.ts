import * as THREE from 'three';
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js';
import type { EnvSnapshot } from '../core/task/types';
import { BALL_PALETTES, createCheckerTexture, createGridTexture } from './ballStyle';
import { CameraController } from './cameraController';

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
  private ball!: THREE.Mesh;
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

    const bg = new THREE.Color('#1b2130');
    this.scene.background = bg;
    this.scene.fog = new THREE.Fog(bg, 28, 62);

    // Image-based light for soft reflections on the ball; kept subtle.
    const pmrem = new THREE.PMREMGenerator(this.renderer);
    this.scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    this.scene.environmentIntensity = 0.3;
    pmrem.dispose();

    this.scene.add(new THREE.HemisphereLight(0xdfe9ff, 0x3b3a40, 0.32));

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
    this.controls = new CameraController(this.camera, this.renderer.domElement);

    this.resizeObs = new ResizeObserver(() => this.resize());
    this.resizeObs.observe(container);
    this.applyTier('high');
    this.resize();
  }

  /** Build floor, target tile and ball for a task's geometry. */
  buildTask(g: Record<string, number>): void {
    this.world.clear();
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
    this.epoch = -1;
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
      this.ball.position.lerp(this.snapPos, k);
      this.ball.quaternion.slerp(this.snapQuat, k);
      this.targetGroup.position.lerp(this.snapTarget, 1 - Math.exp(-30 * dt));
      this.controls.update(dt, this.ball.position);
      if (this.flash > 0) {
        this.flash = Math.max(0, this.flash - dt * 2.2);
        this.targetMat.emissiveIntensity = 0.12 + this.flash * 1.6;
      }
    }
    this.renderer.render(this.scene, this.camera);

    this.autoAdapt(now, dtMs);
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
