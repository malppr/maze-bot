import { mountMazeDemo } from '../src';
import { VERSIONS } from '../src/demo/versions';
// Dev only: the site's mascot (not part of this MIT package).
import sprite from '../../../malppr.github.io/src/assets/mascot/mascot-top.svg?url';

// Failure versions for recording the write-up clips (?v=v0-moonwalker / ?v=v2-scaredy). Never shipped to the site.
VERSIONS.push(
  {
    id: 'v0-moonwalker',
    name: 'Moonwalker',
    blurb: 'Learned to drive backwards, blind.',
    hidden: true,
    load: () => import('../../artifacts/release/versions/v0-moonwalker/weights.json'),
  },
  {
    id: 'v2-scaredy',
    name: 'Scaredy-Wheely',
    blurb: 'Too scared of walls to move.',
    hidden: true,
    load: () => import('../../artifacts/release/versions/v2-scaredy/weights.json'),
  },
);

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
