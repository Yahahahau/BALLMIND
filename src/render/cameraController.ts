import * as THREE from 'three';

/**
 * Smooth follow camera with touch/mouse orbit.
 *  - one finger / left mouse drag: orbit
 *  - two fingers pinch / wheel: zoom
 */
export class CameraController {
  yaw = 0;
  pitch = 0.95;
  dist = 18;
  /** 0.65 keeps Reach The Square's arena framed; 1 follows the ball exactly (Obby). */
  followScale = 0.65;
  private overview = false;
  private ovCenter = new THREE.Vector3();
  private ovSize = 30;
  private readonly focus = new THREE.Vector3(0, 0.3, 0);
  private readonly desired = new THREE.Vector3();
  private readonly pointers = new Map<number, { x: number; y: number }>();
  private pinchStart = 0;
  private distStart = 0;
  private static readonly DEFAULT = { yaw: 0, pitch: 0.95, dist: 18 };

  constructor(
    readonly camera: THREE.PerspectiveCamera,
    el: HTMLElement,
  ) {
    el.style.touchAction = 'none';
    el.addEventListener('pointerdown', this.down);
    el.addEventListener('pointermove', this.move);
    el.addEventListener('pointerup', this.up);
    el.addEventListener('pointercancel', this.up);
    el.addEventListener('wheel', this.wheel, { passive: false });
    el.addEventListener('dblclick', () => this.reset());
  }

  setOverview(on: boolean, center: THREE.Vector3, size: number): void {
    this.overview = on;
    this.ovCenter.copy(center);
    this.ovSize = size;
  }

  reset(): void {
    Object.assign(this, CameraController.DEFAULT);
    if (this.followScale === 1) this.dist = 13;
  }

  private down = (e: PointerEvent): void => {
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    this.pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (this.pointers.size === 2) {
      this.pinchStart = this.pinchDistance();
      this.distStart = this.dist;
    }
  };

  private move = (e: PointerEvent): void => {
    const p = this.pointers.get(e.pointerId);
    if (!p) return;
    const dx = e.clientX - p.x;
    const dy = e.clientY - p.y;
    p.x = e.clientX;
    p.y = e.clientY;
    if (this.pointers.size === 1) {
      this.yaw -= dx * 0.007;
      this.pitch = THREE.MathUtils.clamp(this.pitch + dy * 0.006, 0.2, 1.5);
    } else if (this.pointers.size === 2 && this.pinchStart > 0) {
      this.dist = THREE.MathUtils.clamp((this.distStart * this.pinchStart) / Math.max(20, this.pinchDistance()), 5, 40);
    }
  };

  private up = (e: PointerEvent): void => {
    this.pointers.delete(e.pointerId);
    this.pinchStart = 0;
  };

  private wheel = (e: WheelEvent): void => {
    e.preventDefault();
    this.dist = THREE.MathUtils.clamp(this.dist * Math.exp(e.deltaY * 0.0012), 5, 40);
  };

  private pinchDistance(): number {
    const [a, b] = [...this.pointers.values()];
    return Math.hypot(a.x - b.x, a.y - b.y);
  }

  /** `follow` is the point to track (the ball). Call once per rendered frame. */
  update(dt: number, follow: THREE.Vector3): void {
    if (!Number.isFinite(this.focus.x + this.focus.y + this.focus.z)) this.focus.copy(Number.isFinite(follow.x + follow.y + follow.z) ? follow : this.ovCenter);
    if (this.overview) {
      this.desired.copy(this.ovCenter);
      this.focus.lerp(this.desired, 1 - Math.exp(-4 * dt));
      this.camera.position.lerp(
        new THREE.Vector3(this.focus.x + 0.001, this.focus.y + this.ovSize * 1.3 + 6, this.focus.z + 0.001 + this.ovSize * 0.12),
        1 - Math.exp(-4 * dt),
      );
      this.camera.lookAt(this.focus);
      return;
    }
    const k = this.followScale;
    this.desired.set(follow.x * k, follow.y * (k === 1 ? 1 : 0.6), follow.z * k);
    this.focus.lerp(this.desired, 1 - Math.exp(-(k === 1 ? 5 : 3.5) * dt));
    const cp = Math.cos(this.pitch);
    this.camera.position.set(
      this.focus.x + Math.sin(this.yaw) * cp * this.dist,
      this.focus.y + Math.sin(this.pitch) * this.dist,
      this.focus.z + Math.cos(this.yaw) * cp * this.dist,
    );
    this.camera.lookAt(this.focus);
  }
}
