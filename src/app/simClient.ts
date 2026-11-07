import type { FromWorker, ToWorker } from '../worker/protocol';
// Vite bundles the worker (and Rapier's WASM) inline so the app stays a single file.
import SimWorker from '../worker/sim.worker?worker&inline';

/** Typed bridge to the simulation worker. */
export class SimClient {
  private worker: Worker;
  onMessage: (m: FromWorker) => void = () => undefined;

  constructor() {
    this.worker = new SimWorker();
    this.worker.onmessage = (e: MessageEvent<FromWorker>) => this.onMessage(e.data);
    this.worker.onerror = (e) => this.onMessage({ type: 'error', message: e.message || 'Worker error' });
  }

  send(m: ToWorker): void {
    this.worker.postMessage(m);
  }

  terminate(): void {
    this.worker.terminate();
  }
}
