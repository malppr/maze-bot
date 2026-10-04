import { mountMazeDemo } from '../src';
// Dev only: the site's mascot (not part of this MIT package).
import sprite from '../../../malppr.github.io/src/assets/mascot/mascot-top.svg?url';

const q = new URLSearchParams(location.search);
const el = document.getElementById('demo')!;
const handle = mountMazeDemo(el, {
  mode: q.get('mode') === 'draw' ? 'draw' : 'obstacles',
  showNetwork: q.get('network') !== '0',
  version: q.get('v') ?? undefined,
  sprite,
});
(window as unknown as { demo: typeof handle }).demo = handle;

const root = document.documentElement;
if (q.get('theme') === 'dark') root.dataset.theme = 'dark';
document.getElementById('theme')!.onclick = () => {
  root.dataset.theme = root.dataset.theme === 'dark' ? 'light' : 'dark';
};
