import type { ReplayRecord } from '../core/training/model';

const RECENT_PER_STAGE = 12;
const PERIODIC_MAX = 40;
const MILESTONE_MAX = 80;

/**
 * Keeps a bounded, useful set of episode recordings on the main thread:
 * the first attempt and every "new best" milestone per stage (to compare old vs new),
 * a sparse periodic sample, and the most recent attempts. A trace is ~6 bytes per decision.
 */
export class ReplayStore {
  records: ReplayRecord[] = [];
  version = 0;

  set(list: ReplayRecord[]): void {
    this.records = list.slice().sort((a, b) => a.ep - b.ep);
    this.version++;
  }

  clear(): void {
    this.records = [];
    this.version++;
  }

  add(list: ReplayRecord[]): void {
    if (!list.length) return;
    this.records.push(...list);
    this.prune();
    this.version++;
  }

  private prune(): void {
    const r = this.records;
    // recent: only the newest N per stage (earlier ones are dropped unless they are first/milestone/periodic)
    const keep = new Set<ReplayRecord>();
    const perStage = new Map<number, ReplayRecord[]>();
    for (const x of r) {
      if (!perStage.has(x.stage)) perStage.set(x.stage, []);
      perStage.get(x.stage)!.push(x);
    }
    for (const list of perStage.values()) for (const x of list.slice(-RECENT_PER_STAGE)) keep.add(x);
    const mil = r.filter((x) => x.kind === 'first' || x.kind === 'milestone');
    for (const x of mil.slice(-MILESTONE_MAX)) keep.add(x);
    // keep every stage's very first record
    for (const list of perStage.values()) if (list[0]) keep.add(list[0]);
    const per = r.filter((x) => x.kind === 'periodic');
    for (const x of per.slice(-PERIODIC_MAX)) keep.add(x);
    this.records = r.filter((x) => keep.has(x));
  }

  byEp(ep: number): ReplayRecord | undefined {
    return this.records.find((x) => x.ep === ep);
  }
}
