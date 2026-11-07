import * as THREE from 'three';

export interface BallPalette {
  name: string;
  a: string;
  b: string;
}

export const BALL_PALETTES: BallPalette[] = [
  { name: 'Teal', a: '#14b8a6', b: '#f3f6f8' },
  { name: 'Coral', a: '#ff5d73', b: '#fff1e6' },
  { name: 'Cobalt', a: '#4f7cff', b: '#f5f7ff' },
  { name: 'Amber', a: '#ffc233', b: '#2b2f3a' },
  { name: 'Violet', a: '#9b6bff', b: '#f4f0ff' },
  { name: 'Lime', a: '#7bd33b', b: '#17331c' },
];

/** Procedural checkerboard (16 × 8 cells) for the sphere's equirectangular UVs. */
export function createCheckerTexture(p: BallPalette, anisotropy: number): THREE.CanvasTexture {
  const W = 1024;
  const H = 512;
  const cols = 16;
  const rows = 8;
  const c = document.createElement('canvas');
  c.width = W;
  c.height = H;
  const g = c.getContext('2d')!;
  const cw = W / cols;
  const ch = H / rows;
  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < cols; x++) {
      g.fillStyle = (x + y) % 2 === 0 ? p.a : p.b;
      g.fillRect(x * cw, y * ch, cw + 1, ch + 1);
    }
  }
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = anisotropy;
  tex.wrapS = THREE.RepeatWrapping;
  tex.needsUpdate = true;
  return tex;
}

/** Faint 1 m grid for the floor so rolling motion and distances are readable. */
export function createGridTexture(cells: number, anisotropy: number): THREE.CanvasTexture {
  const S = 1024;
  const c = document.createElement('canvas');
  c.width = S;
  c.height = S;
  const g = c.getContext('2d')!;
  g.fillStyle = '#cfcac0';
  g.fillRect(0, 0, S, S);
  g.strokeStyle = 'rgba(60,55,48,0.16)';
  g.lineWidth = 2;
  const step = S / cells;
  for (let i = 0; i <= cells; i++) {
    const v = Math.min(S - 1, Math.round(i * step));
    g.beginPath();
    g.moveTo(v, 0);
    g.lineTo(v, S);
    g.moveTo(0, v);
    g.lineTo(S, v);
    g.stroke();
  }
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = anisotropy;
  tex.needsUpdate = true;
  return tex;
}
