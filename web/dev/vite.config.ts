import { defineConfig } from 'vite';

// Standalone dev page for the demo: `npm run dev`. The site's mascot sprite is read from the sibling repo.
export default defineConfig({
  root: import.meta.dirname,
  server: { port: 5180, fs: { allow: ['../..', '../../../malppr.github.io/src/assets'] } },
});
