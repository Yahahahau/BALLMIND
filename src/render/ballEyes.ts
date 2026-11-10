import * as THREE from 'three';

/**
 * Two cartoon eyes for the AI ball. PURELY visual: not a child of the physics-driven ball mesh,
 * no collider, no shadows. The "face" group follows the ball's position and keeps a stable facing
 * (the ball's travel heading), so the eyes never roll with the ball. Pupils are placed from the AI
 * camera's real forward vector, expressed in the face's local frame (no scripted/random animation).
 */
export class BallEyes {
  readonly group = new THREE.Group();
  private face = new THREE.Group();
  private pupils: THREE.Mesh[] = [];
  private glints: THREE.Mesh[] = [];
  private geos: THREE.BufferGeometry[] = [];
  private mats: THREE.Material[] = [];
  private eyeR: number;
  private heading = 0; // radians, local +Z of the face in world XZ: (sin, cos)
  private headingKnown = false;
  private gl = new THREE.Vector3(0, 0, 1); // smoothed gaze in face-local space
  private last = new THREE.Vector3();
  private tmp = new THREE.Vector3();
  private zAxis = new THREE.Vector3(0, 0, 1);
  private hasLast = false;
  /** Max pupil excursion from the eye centre, radians (keeps the pupil inside the visible eyeball). */
  private readonly maxPupil = (36 * Math.PI) / 180;

  constructor(private ballRadius: number) {
    const R = ballRadius;
    this.eyeR = R * 0.25;
    const sclera = new THREE.MeshStandardMaterial({ color: '#fbfbff', roughness: 0.22, metalness: 0 });
    const pupilM = new THREE.MeshStandardMaterial({ color: '#0a0c12', roughness: 0.12, metalness: 0.1 });
    const glintM = new THREE.MeshBasicMaterial({ color: '#ffffff' });
    this.mats.push(sclera, pupilM, glintM);
    const scG = new THREE.SphereGeometry(this.eyeR, 24, 18);
    const pG = new THREE.SphereGeometry(this.eyeR * 0.56, 20, 14);
    const gG = new THREE.SphereGeometry(this.eyeR * 0.11, 8, 6);
    this.geos.push(scG, pG, gG);
    const az = (21 * Math.PI) / 180;
    const el = (9 * Math.PI) / 180;
    for (const s of [-1, 1]) {
      const eye = new THREE.Group();
      const dir = new THREE.Vector3(Math.sin(az) * s * Math.cos(el), Math.sin(el), Math.cos(az) * Math.cos(el));
      // centre slightly inside the ball surface so the eyeball sits in the ball, not floating
      eye.position.copy(dir).multiplyScalar(R * 0.93);
      eye.add(new THREE.Mesh(scG, sclera));
      const pupil = new THREE.Mesh(pG, pupilM);
      pupil.scale.set(1, 1, 0.3); // flattened, rounded disc
      const glint = new THREE.Mesh(gG, glintM);
      eye.add(pupil, glint);
      this.face.add(eye);
      this.pupils.push(pupil);
      this.glints.push(glint);
    }
    this.group.add(this.face);
    this.place(0, 0, 1, 1);
  }

  /** Put the pupils (and their glints) on the eyeball surface along local direction (x,y,z). */
  private place(x: number, y: number, z: number, _k: number): void {
    const r = this.eyeR;
    for (let i = 0; i < 2; i++) {
      const p = this.pupils[i];
      p.position.set(x * r * 0.93, y * r * 0.93, z * r * 0.93);
      this.tmp.set(x, y, z).normalize();
      p.quaternion.setFromUnitVectors(this.zAxis, this.tmp); // flat side faces outward
      const g = this.glints[i];
      // glint stays fixed up-and-aside of the pupil on the eyeball
      const gx = x * 0.93 + 0.28;
      const gy = y * 0.93 + 0.3;
      const gz = Math.sqrt(Math.max(0.05, 1 - gx * gx - gy * gy));
      g.position.set(gx * r, gy * r, gz * r * 1.0);
    }
  }

  /**
   * @param pos   ball position (world)
   * @param gazeWorld AI camera forward vector in world space, or null when the task has no AI camera
   *                  (eyes then look straight ahead).
   * @param headingHint horizontal facing used before the ball has moved (camera heading for vision tasks).
   */
  update(dt: number, pos: THREE.Vector3, gazeWorld: THREE.Vector3 | null, headingHint?: number, fp?: { yaw: number; pitch: number } | null): void {
    this.group.position.copy(pos);
    if (fp) {
      this.updateFp(dt, fp);
      return;
    }
    this.face.rotation.x = 0;
    // heading follows the (visual) travel direction; ignore teleports (respawn / replay seek)
    if (this.hasLast && dt > 0) {
      const dx = pos.x - this.last.x;
      const dz = pos.z - this.last.z;
      const d = Math.hypot(dx, dz);
      if (d < 3 && d / dt > 0.35) {
        const target = Math.atan2(dx, dz);
        let diff = target - this.heading;
        diff = Math.atan2(Math.sin(diff), Math.cos(diff));
        this.heading += diff * (1 - Math.exp(-6 * dt));
        this.headingKnown = true;
      }
    }
    this.last.copy(pos);
    this.hasLast = true;
    if (!this.headingKnown && headingHint !== undefined) this.heading = headingHint;
    this.face.rotation.y = this.heading;

    // AI camera forward → face-local frame (rotate by -heading about Y)
    let tx = 0;
    let ty = 0;
    let tz = 1;
    if (gazeWorld) {
      const c = Math.cos(this.heading);
      const s = Math.sin(this.heading);
      tx = gazeWorld.x * c - gazeWorld.z * s;
      tz = gazeWorld.x * s + gazeWorld.z * c;
      ty = gazeWorld.y;
    }
    const k = 1 - Math.exp(-16 * Math.max(0, dt));
    this.gl.x += (tx - this.gl.x) * k;
    this.gl.y += (ty - this.gl.y) * k;
    this.gl.z += (tz - this.gl.z) * k;
    const g = this.gl;
    const len = Math.hypot(g.x, g.y, g.z) || 1;
    const nx = g.x / len;
    const ny = g.y / len;
    const nz = g.z / len;
    // angle of the gaze away from straight-ahead → pupil excursion on the eyeball (clamped)
    const theta = Math.acos(Math.min(1, Math.max(-1, nz)));
    const a = this.maxPupil * Math.min(1, theta / (Math.PI / 3));
    const h = Math.hypot(nx, ny);
    const ux = h > 1e-4 ? nx / h : 0;
    const uy = h > 1e-4 ? ny / h : 0;
    this.place(Math.sin(a) * ux, Math.sin(a) * uy, Math.cos(a), 1);
  }

  private faceYaw = 0; // degrees (camera convention) the face rig is currently turned to
  private facePitch = 0;
  private fpInit = false;

  /**
   * First-person gaze: the AI camera's yaw/pitch is the source of truth. The face rig turns towards it
   * (smoothly), the eyeballs ride on the face, and the pupils show the remaining difference between where
   * the camera points and where the face currently points – so they lead every gaze change and settle to
   * the centre once the face has caught up. Nothing here is random or independent of the camera.
   */
  private updateFp(dt: number, g: { yaw: number; pitch: number }): void {
    if (!this.fpInit) {
      this.faceYaw = g.yaw;
      this.facePitch = g.pitch;
      this.fpInit = true;
    }
    const wrap = (d: number) => ((((d + 180) % 360) + 360) % 360) - 180;
    const dyaw = wrap(g.yaw - this.faceYaw);
    const dpit = g.pitch - this.facePitch;
    const k = 1 - Math.exp(-9 * Math.max(0, dt));
    this.faceYaw = wrap(this.faceYaw + dyaw * k);
    this.facePitch += dpit * k;
    // face rig: front (+Z local) points along the camera direction (yaw 0 = world -Z)
    this.face.rotation.order = 'YXZ';
    this.face.rotation.y = Math.PI - (this.faceYaw * Math.PI) / 180;
    this.face.rotation.x = -(this.facePitch * Math.PI) / 180;
    // residual gaze (what the camera does beyond the face) → pupil direction in face space
    const ry = (wrap(g.yaw - this.faceYaw) * Math.PI) / 180;
    const rp = ((g.pitch - this.facePitch) * Math.PI) / 180;
    // camera yaw to the right ⇒ looking toward the AI's right = local −X (front is +Z)
    const lx = -Math.sin(ry) * Math.cos(rp);
    const ly = Math.sin(rp);
    const lz = Math.cos(ry) * Math.cos(rp);
    this.gl.set(lx, ly, lz);
    const theta = Math.acos(Math.min(1, Math.max(-1, lz)));
    const a = this.maxPupil * Math.min(1, theta / ((20 * Math.PI) / 180));
    const h = Math.hypot(lx, ly);
    this.place(h > 1e-5 ? (Math.sin(a) * lx) / h : 0, h > 1e-5 ? (Math.sin(a) * ly) / h : 0, Math.cos(a), 1);
  }

  /** Pupil offsets (face-local, unit eyeball) for tests/inspection. */
  pupilDirection(): [number, number, number] {
    const p = this.pupils[0].position;
    const r = this.eyeR * 0.93;
    return [p.x / r, p.y / r, p.z / r];
  }

  dispose(): void {
    this.group.removeFromParent();
    for (const g of this.geos) g.dispose();
    for (const m of this.mats) m.dispose();
  }
}
