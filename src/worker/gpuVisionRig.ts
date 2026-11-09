import * as THREE from 'three';
import type { EnvSnapshot } from '../core/task/types';
import { REACH_CONFIG as C } from '../core/tasks/reachSquare/config';
import { visionCameraPose } from '../core/vision/camera';
import type { VisionSettings } from '../core/vision/config';
import { VISION_COLORS, type VisionRig } from '../core/vision/rig';
import { SoftReachRig } from '../core/vision/softRig';

const rgb = (c: [number, number, number]) => new THREE.Color().setRGB(c[0] / 255, c[1] / 255, c[2] / 255, THREE.LinearSRGBColorSpace);

/**
 * The AI camera, rendered by WebGL2 on an OffscreenCanvas inside the simulation worker. It draws the
 * real scene geometry with flat unlit materials into a tiny render target and reads the RGB pixels back.
 * It shares nothing with the player's renderer: no shadows, no GTAO, no sky, no tone mapping.
 */
export class GpuReachRig implements VisionRig {
  readonly kind = 'gpu';
  settings: VisionSettings;
  private renderer: THREE.WebGLRenderer;
  private scene = new THREE.Scene();
  private camera = new THREE.PerspectiveCamera(60, 1, 0.1, 120);
  private rt: THREE.WebGLRenderTarget | null = null;
  private ball: THREE.Mesh;
  private tile: THREE.Mesh;
  private px = new Uint8Array(0);
  private disposables: { dispose(): void }[] = [];

  constructor(settings: VisionSettings) {
    this.settings = settings;
    if (typeof OffscreenCanvas === 'undefined') throw new Error('OffscreenCanvas unavailable');
    const canvas = new OffscreenCanvas(settings.size, settings.size);
    this.renderer = new THREE.WebGLRenderer({ canvas: canvas as unknown as HTMLCanvasElement, antialias: false, alpha: false, powerPreference: 'low-power', depth: true });
    this.renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
    this.renderer.toneMapping = THREE.NoToneMapping;
    this.renderer.shadowMap.enabled = false;
    this.renderer.setClearColor(rgb(VISION_COLORS.void), 1);
    this.scene.matrixWorldAutoUpdate = true;
    const mat = (c: [number, number, number]) => {
      const m = new THREE.MeshBasicMaterial({ color: rgb(c) });
      this.disposables.push(m);
      return m;
    };
    const floorG = new THREE.BoxGeometry(C.arenaHalf * 2, C.floorThickness, C.arenaHalf * 2);
    const tileG = new THREE.BoxGeometry(C.targetHalf * 2, C.tileHeight, C.targetHalf * 2);
    const ballG = new THREE.SphereGeometry(C.ballRadius, 20, 14);
    this.disposables.push(floorG, tileG, ballG);
    const floor = new THREE.Mesh(floorG, mat(VISION_COLORS.floor));
    floor.position.y = -C.floorThickness / 2;
    this.tile = new THREE.Mesh(tileG, mat(VISION_COLORS.target));
    this.ball = new THREE.Mesh(ballG, mat(VISION_COLORS.ball));
    this.scene.add(floor, this.tile, this.ball);
    this.camera.matrixAutoUpdate = false;
    this.camera.matrixWorldAutoUpdate = false;
    this.configure(settings);
  }

  configure(s: VisionSettings): void {
    this.settings = s;
    this.rt?.dispose();
    this.rt = new THREE.WebGLRenderTarget(s.size, s.size, {
      format: THREE.RGBAFormat,
      type: THREE.UnsignedByteType,
      depthBuffer: true,
      generateMipmaps: false,
      minFilter: THREE.NearestFilter,
      magFilter: THREE.NearestFilter,
    });
    this.px = new Uint8Array(s.size * s.size * 4);
    this.camera.fov = s.fov;
    this.camera.updateProjectionMatrix();
  }

  render(snap: EnvSnapshot, out: Uint8Array): void {
    const n = this.settings.size;
    const b = snap.bodies.ball.p;
    const t = snap.bodies.target.p;
    this.ball.position.set(b[0], b[1], b[2]);
    this.tile.position.set(t[0], C.tileHeight / 2, t[2]);
    const cam = visionCameraPose(this.settings, b[0], b[1], b[2]);
    this.camera.matrixWorld.makeBasis(new THREE.Vector3(...cam.right), new THREE.Vector3(...cam.up), new THREE.Vector3(-cam.fwd[0], -cam.fwd[1], -cam.fwd[2]));
    this.camera.matrixWorld.setPosition(cam.eye[0], cam.eye[1], cam.eye[2]);
    this.camera.matrixWorldInverse.copy(this.camera.matrixWorld).invert();
    this.renderer.setRenderTarget(this.rt);
    this.renderer.render(this.scene, this.camera);
    this.renderer.readRenderTargetPixels(this.rt!, 0, 0, n, n, this.px);
    this.renderer.setRenderTarget(null);
    const plane = n * n;
    for (let j = 0; j < n; j++) {
      const src = (n - 1 - j) * n * 4; // GL rows run bottom-up
      for (let i = 0; i < n; i++) {
        const p = j * n + i;
        const s = src + i * 4;
        out[p] = this.px[s];
        out[plane + p] = this.px[s + 1];
        out[2 * plane + p] = this.px[s + 2];
      }
    }
  }

  dispose(): void {
    this.rt?.dispose();
    for (const d of this.disposables) d.dispose();
    this.renderer.dispose();
    this.renderer.forceContextLoss();
  }
}

/** GPU rig when WebGL2 + OffscreenCanvas work in this worker, otherwise the (slower) CPU camera. */
export function createVisionRig(settings: VisionSettings): { rig: VisionRig; note?: string } {
  try {
    return { rig: new GpuReachRig(settings) };
  } catch (e) {
    return { rig: new SoftReachRig(settings), note: `AI camera runs on the CPU (WebGL2 in worker unavailable: ${(e as Error).message})` };
  }
}
