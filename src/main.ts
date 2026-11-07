import './ui/styles.css';
import { App } from './ui/app';

const root = document.getElementById('app')!;
new App(root).boot().catch((err) => {
  console.error(err);
  root.innerHTML = `<pre style="padding:24px;color:#fb7185;white-space:pre-wrap">BALLMIND failed to start:\n${String(err?.message ?? err)}</pre>`;
});
