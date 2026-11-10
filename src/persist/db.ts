import type { ModelBlob, ReplayRecord } from '../core/training/model';

/** Minimal IndexedDB key/value store. Every call degrades gracefully if storage is unavailable. */
const DB_NAME = 'ballmind';
const STORE = 'kv';

function open(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    if (typeof indexedDB === 'undefined') return reject(new Error('IndexedDB unavailable'));
    const req = indexedDB.open(DB_NAME, 1);
    req.onupgradeneeded = () => req.result.createObjectStore(STORE);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function get<T>(key: string): Promise<T | undefined> {
  const db = await open();
  return new Promise((resolve, reject) => {
    const r = db.transaction(STORE).objectStore(STORE).get(key);
    r.onsuccess = () => resolve(r.result as T | undefined);
    r.onerror = () => reject(r.error);
  });
}

async function put(key: string, value: unknown): Promise<void> {
  const db = await open();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, 'readwrite');
    tx.objectStore(STORE).put(value, key);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

async function del(key: string): Promise<void> {
  const db = await open();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, 'readwrite');
    tx.objectStore(STORE).delete(key);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

const modelKey = (taskId: string) => `model:${taskId}`;

export const saveModel = (m: ModelBlob) => put(modelKey(m.taskId), m);
export const deleteModel = (taskId: string) => del(modelKey(taskId));
export async function loadModel(taskId: string): Promise<ModelBlob | undefined> {
  try {
    return await get<ModelBlob>(modelKey(taskId));
  } catch {
    return undefined;
  }
}

export interface Settings {
  speed: number;
  quality: 'auto' | 'high' | 'medium' | 'low';
  palette: number;
  gtao?: import('../render/gtao').GtaoSettings;
  music?: import('../audio/musicPlayer').MusicSettings;
  vision?: import('../core/vision/config').VisionSettings;
  visionFp?: import('../core/vision/config').VisionSettings;
  sky?: { source: 'default' | 'custom' | 'none'; values: import('../render/skybox').SkySettings };
}

export async function loadSettings(): Promise<Partial<Settings>> {
  try {
    return (await get<Partial<Settings>>('settings')) ?? {};
  } catch {
    return {};
  }
}

export const saveSettings = (s: Settings) => put('settings', s).catch(() => undefined);

const replayKey = (taskId: string) => `replays:${taskId}`;
export const saveReplays = (taskId: string, r: ReplayRecord[]) => put(replayKey(taskId), r).catch(() => undefined);
export const deleteReplays = (taskId: string) => del(replayKey(taskId)).catch(() => undefined);
export async function loadReplays(taskId: string): Promise<ReplayRecord[]> {
  try {
    return (await get<ReplayRecord[]>(replayKey(taskId))) ?? [];
  } catch {
    return [];
  }
}

/** The imported skybox files (kept local in IndexedDB so the sky survives reloads). */
export const saveSky = (s: import('../render/skybox').StoredSky) => put('skybox', s).catch(() => undefined);
export const deleteSky = () => del('skybox').catch(() => undefined);
export async function loadSky(): Promise<import('../render/skybox').StoredSky | undefined> {
  try {
    return await get('skybox');
  } catch {
    return undefined;
  }
}

/**
 * Before the first-person Vision Agent V2 exists in a browser profile, copy every legacy saved model (and replay
 * set) to a separate `backup:v1:*` key. Never overwrites an existing backup, never touches the originals.
 */
export async function backupLegacyModels(): Promise<{ backedUp: string[]; existing: string[] }> {
  const res = { backedUp: [] as string[], existing: [] as string[] };
  for (const id of ['reach-square', 'reach-vision', 'obby-challenge']) {
    const m = await get<ModelBlob>(modelKey(id)).catch(() => undefined);
    if (!m) continue;
    const bk = `backup:v1:${modelKey(id)}`;
    if (await get(bk).catch(() => undefined)) {
      res.existing.push(id);
      continue;
    }
    await put(bk, m).catch(() => undefined);
    const rp = await get(replayKey(id)).catch(() => undefined);
    if (rp) await put(`backup:v1:${replayKey(id)}`, rp).catch(() => undefined);
    res.backedUp.push(id);
  }
  return res;
}

/** Put a backed-up legacy model back (explicit user action). */
export async function restoreLegacyBackup(taskId: string): Promise<boolean> {
  const m = await get<ModelBlob>(`backup:v1:${modelKey(taskId)}`).catch(() => undefined);
  if (!m) return false;
  await put(modelKey(taskId), m);
  const rp = await get(`backup:v1:${replayKey(taskId)}`).catch(() => undefined);
  if (rp) await put(replayKey(taskId), rp);
  return true;
}
