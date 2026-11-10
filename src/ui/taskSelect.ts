import type { TaskInfo } from '../core/task/types';

/** Mission preview illustrations, keyed by task id. */
const PREVIEWS: Record<string, string> = {
  'reach-square': `
<svg viewBox="0 0 320 200" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Ball rolling toward the target square">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#232b3d"/><stop offset="1" stop-color="#151b27"/></linearGradient>
    <radialGradient id="ballg" cx="35%" cy="30%" r="75%"><stop offset="0" stop-color="#fff" stop-opacity=".55"/><stop offset=".45" stop-color="#fff" stop-opacity="0"/><stop offset="1" stop-color="#000" stop-opacity=".35"/></radialGradient>
    <clipPath id="bc"><circle cx="78" cy="118" r="20"/></clipPath>
  </defs>
  <rect width="320" height="200" fill="url(#bg)"/>
  <polygon points="160,38 296,100 160,168 24,100" fill="#e6e2da"/>
  <polygon points="24,100 160,168 160,180 24,112" fill="#bdb8ae"/>
  <polygon points="160,168 296,100 296,112 160,180" fill="#a8a399"/>
  <g stroke="#4a4338" stroke-opacity=".14" stroke-width="1">
    <path d="M70,60 L206,128 M115,49 L251,117 M25,100 L161,168 M160,38 L296,100"/>
    <path d="M250,60 L114,128 M205,49 L69,117 M295,100 L159,168 M160,38 L24,100"/>
  </g>
  <polygon points="214,66 244,80 214,94 184,80" fill="#3a2c24"/>
  <polygon points="214,62 242,75 214,88 186,75" fill="#ff7a2f"/>
  <path d="M104,112 C140,100 160,92 180,84" stroke="#2dd4bf" stroke-width="2.5" stroke-dasharray="2 7" stroke-linecap="round" fill="none"/>
  <path d="M176,79 l9,5 l-10,3z" fill="#2dd4bf"/>
  <ellipse cx="82" cy="134" rx="20" ry="6" fill="#000" opacity=".22"/>
  <circle cx="78" cy="118" r="20" fill="#14b8a6"/>
  <g clip-path="url(#bc)" fill="#f3f6f8">
    <rect x="58" y="98" width="10" height="10"/><rect x="78" y="98" width="10" height="10"/>
    <rect x="68" y="108" width="10" height="10"/><rect x="88" y="108" width="10" height="10"/>
    <rect x="58" y="118" width="10" height="10"/><rect x="78" y="118" width="10" height="10"/>
    <rect x="68" y="128" width="10" height="10"/><rect x="88" y="128" width="10" height="10"/>
  </g>
  <circle cx="78" cy="118" r="20" fill="url(#ballg)"/>
</svg>`,
};

PREVIEWS['reach-vision'] = PREVIEWS['reach-square'];
PREVIEWS['reach-fp'] = PREVIEWS['reach-square'];

/** Modal "SELECT TRAINING TASK". Rendered from the task catalog, so new tasks appear automatically. */
export function showTaskSelect(tasks: TaskInfo[], onPick: (t: TaskInfo) => void): void {
  const ov = document.createElement('div');
  ov.className = 'overlay';
  ov.innerHTML = `
    <div class="modal" role="dialog" aria-modal="true" aria-label="Select training task">
      <h1>SELECT TRAINING TASK</h1>
      <div class="sub">Choose a mission. The AI learns it on its own, by trial and error, inside real physics.</div>
      <div id="cards"></div>
    </div>`;
  const cards = ov.querySelector('#cards')!;
  for (const t of tasks) {
    const b = document.createElement('button');
    b.className = 'task-card';
    b.innerHTML = `
      <div>${PREVIEWS[t.id] ?? ''}</div>
      <div>
        <h3>${t.title}</h3>
        <p>${t.description}</p>
        <span class="tag">Deep Q-Network</span><span class="tag">${t.actionLabels.length} actions</span><span class="tag">${t.observationSize} inputs</span>
        <div class="label-sm">Tap to start ▸</div>
      </div>`;
    b.addEventListener('click', () => {
      ov.remove();
      onPick(t);
    });
    cards.appendChild(b);
  }
  document.body.appendChild(ov);
}

export function confirmDialog(title: string, text: string, okLabel: string): Promise<boolean> {
  return new Promise((resolve) => {
    const ov = document.createElement('div');
    ov.className = 'overlay';
    ov.style.zIndex = '60';
    ov.innerHTML = `
      <div class="modal" style="width:min(440px,100%)" role="alertdialog" aria-modal="true">
        <h1>${title}</h1>
        <div class="sub" style="margin:10px 0 18px">${text}</div>
        <div class="row">
          <button class="btn" data-v="0">Cancel</button>
          <button class="btn danger" data-v="1">${okLabel}</button>
        </div>
      </div>`;
    ov.addEventListener('click', (e) => {
      const v = (e.target as HTMLElement).closest('button')?.dataset.v;
      if (v === undefined) return;
      ov.remove();
      resolve(v === '1');
    });
    document.body.appendChild(ov);
  });
}

export function toast(msg: string, isError = false, ms = 2800): void {
  const el = document.createElement('div');
  el.className = 'toast' + (isError ? ' err' : '');
  el.textContent = msg;
  document.body.appendChild(el);
  setTimeout(() => el.remove(), ms);
}
