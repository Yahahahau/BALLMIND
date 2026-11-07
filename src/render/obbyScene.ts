import * as THREE from 'three';
import { STAGES } from '../core/tasks/obby/levels/compile';
import { hazardCenter, hazardRotation, platformCenter, platformRotation } from '../core/tasks/obby/kinematics';
import type { PlatformKind } from '../core/tasks/obby/types';

/** Distinct, calm colours per platform type. */
export const KIND_COLORS: Record<PlatformKind | 'hazard', string> = {
  start: '#34d399',
  normal: '#9aa7bd',
  checkpoint: '#fbbf24',
  moving: '#38bdf8',
  rotating: '#a78bfa',
  slippery: '#67e8f9',
  finish: '#fb7185',
  hazard: '#ef4444',
};

const unitBox = new THREE.BoxGeometry(1, 1, 1);

const kindOf = (k: PlatformKind, moving: boolean, spinning: boolean): PlatformKind =>
  k === 'normal' && spinning ? 'rotating' : k === 'normal' && moving ? 'moving' : k;

/**
 * Renders one Obby stage. Static platforms are drawn with a single InstancedMesh;
 * moving platforms and hazards are individual meshes driven either by the physics
 * snapshot (live) or by the shared pure kinematic functions (replay).
 */
export class ObbyScene {
  readonly group = new THREE.Group();
  stageIdx = -1;
  private disposables: THREE.Material[] = [];
  private dynMeshes: THREE.Mesh[] = [];
  private dynDefs: { kind: 'plat' | 'haz'; i: number }[] = [];
  private props = new THREE.Group();

  setStage(i: number): void {
    if (i === this.stageIdx) return;
    this.clear();
    this.stageIdx = i;
    const st = STAGES[i];
    const mat = new THREE.MeshStandardMaterial({ roughness: 0.62, metalness: 0.02 });
    this.disposables.push(mat);
    const statics = st.def.platforms.filter((p) => !p.motion && !p.spin);
    const inst = new THREE.InstancedMesh(unitBox, mat, Math.max(1, statics.length));
    inst.castShadow = inst.receiveShadow = true;
    const m = new THREE.Matrix4();
    const q = new THREE.Quaternion();
    const s = new THREE.Vector3();
    const pos = new THREE.Vector3();
    const col = new THREE.Color();
    statics.forEach((p, k) => {
      const c = platformCenter(p, 0);
      const r = platformRotation(p, 0);
      pos.set(c[0], c[1], c[2]);
      q.set(r[0], r[1], r[2], r[3]);
      s.set(p.size[0], p.thick, p.size[1]);
      m.compose(pos, q, s);
      inst.setMatrixAt(k, m);
      inst.setColorAt(k, col.set(KIND_COLORS[kindOf(p.kind, false, false)]));
    });
    inst.count = statics.length;
    inst.instanceMatrix.needsUpdate = true;
    if (inst.instanceColor) inst.instanceColor.needsUpdate = true;
    this.group.add(inst);

    st.def.platforms.forEach((p, idx) => {
      if (!p.motion && !p.spin) return;
      const mt = new THREE.MeshStandardMaterial({ color: KIND_COLORS[kindOf(p.kind, !!p.motion, !!p.spin)], roughness: 0.55 });
      this.disposables.push(mt);
      const mesh = new THREE.Mesh(unitBox, mt);
      mesh.scale.set(p.size[0], p.thick, p.size[1]);
      mesh.castShadow = mesh.receiveShadow = true;
      this.group.add(mesh);
      this.dynMeshes.push(mesh);
      this.dynDefs.push({ kind: 'plat', i: idx });
    });
    st.def.hazards.forEach((h, idx) => {
      const mt = new THREE.MeshStandardMaterial({ color: KIND_COLORS.hazard, roughness: 0.5, emissive: '#7f1d1d', emissiveIntensity: 0.35 });
      this.disposables.push(mt);
      const mesh = new THREE.Mesh(unitBox, mt);
      mesh.scale.set(h.size[0], h.size[1], h.size[2]);
      mesh.castShadow = mesh.receiveShadow = true;
      this.group.add(mesh);
      this.dynMeshes.push(mesh);
      this.dynDefs.push({ kind: 'haz', i: idx });
    });
    this.applyTime(0);

    // checkpoint flags + finish arch
    this.props = new THREE.Group();
    const pole = new THREE.MeshStandardMaterial({ color: '#e5e7eb', roughness: 0.5 });
    const flagM = new THREE.MeshStandardMaterial({ color: KIND_COLORS.checkpoint, emissive: '#b45309', emissiveIntensity: 0.4, side: THREE.DoubleSide });
    const archM = new THREE.MeshStandardMaterial({ color: KIND_COLORS.finish, emissive: '#9f1239', emissiveIntensity: 0.3 });
    this.disposables.push(pole, flagM, archM);
    for (const idx of st.checkpointIdx) {
      const p = st.def.platforms[idx];
      const g = new THREE.Group();
      const stick = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.04, 1.8), pole);
      stick.position.y = 0.9;
      const flag = new THREE.Mesh(new THREE.PlaneGeometry(0.8, 0.5), flagM);
      flag.position.set(0.42, 1.5, 0);
      g.add(stick, flag);
      g.position.set(p.pos[0] + Math.cos(p.yaw) * (p.size[0] / 2 - 0.3), p.pos[1], p.pos[2] - Math.sin(p.yaw) * (p.size[0] / 2 - 0.3));
      g.traverse((o) => ((o as THREE.Mesh).castShadow = true));
      this.props.add(g);
    }
    const fp = st.def.platforms[st.finishIdx];
    const arch = new THREE.Group();
    const legL = new THREE.Mesh(new THREE.BoxGeometry(0.25, 2.6, 0.25), archM);
    const legR = legL.clone();
    const top = new THREE.Mesh(new THREE.BoxGeometry(fp.size[0] * 0.8, 0.3, 0.3), archM);
    legL.position.set(-fp.size[0] * 0.4, 1.3, 0);
    legR.position.set(fp.size[0] * 0.4, 1.3, 0);
    top.position.y = 2.6;
    arch.add(legL, legR, top);
    arch.position.set(fp.pos[0], fp.pos[1], fp.pos[2]);
    arch.rotation.y = fp.yaw;
    arch.translateZ(fp.size[1] * 0.2);
    arch.traverse((o) => ((o as THREE.Mesh).castShadow = true));
    this.props.add(arch);
    this.group.add(this.props);
  }

  /** Live: poses straight from the physics bodies ([x,y,z,qx,qy,qz,qw] per dynamic object). */
  applyDyn(d: Float32Array): void {
    for (let k = 0; k < this.dynMeshes.length; k++) {
      const o = k * 7;
      if (o + 6 >= d.length) break;
      const m = this.dynMeshes[k];
      m.position.set(d[o], d[o + 1], d[o + 2]);
      m.quaternion.set(d[o + 3], d[o + 4], d[o + 5], d[o + 6]);
    }
  }

  /** Replay: the same pure functions the physics uses. */
  applyTime(t: number): void {
    const st = STAGES[this.stageIdx];
    if (!st) return;
    for (let k = 0; k < this.dynMeshes.length; k++) {
      const d = this.dynDefs[k];
      const m = this.dynMeshes[k];
      if (d.kind === 'plat') {
        const p = st.def.platforms[d.i];
        const c = platformCenter(p, t);
        const r = platformRotation(p, t);
        m.position.set(c[0], c[1], c[2]);
        m.quaternion.set(r[0], r[1], r[2], r[3]);
      } else {
        const h = st.def.hazards[d.i];
        const c = hazardCenter(h, t);
        const r = hazardRotation(h, t);
        m.position.set(c[0], c[1], c[2]);
        m.quaternion.set(r[0], r[1], r[2], r[3]);
      }
    }
  }

  private clear(): void {
    for (const c of [...this.group.children]) {
      this.group.remove(c);
      if (c instanceof THREE.InstancedMesh) c.dispose();
    }
    this.props.traverse((o) => {
      const g = (o as THREE.Mesh).geometry;
      if (g && g !== unitBox) g.dispose();
    });
    for (const m of this.disposables) m.dispose();
    this.disposables = [];
    this.dynMeshes = [];
    this.dynDefs = [];
  }

  dispose(): void {
    this.clear();
  }
}
