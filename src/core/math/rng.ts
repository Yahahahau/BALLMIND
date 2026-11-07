/** Small seeded PRNG (mulberry32). Deterministic, fast, serialisable state. */
export class Rng {
  private s: number;

  constructor(seed = 1) {
    this.s = seed >>> 0 || 1;
  }

  next(): number {
    let t = (this.s += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  range(a: number, b: number): number {
    return a + (b - a) * this.next();
  }

  int(n: number): number {
    return Math.floor(this.next() * n);
  }

  gaussian(): number {
    const u = Math.max(this.next(), 1e-12);
    const v = this.next();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
}
