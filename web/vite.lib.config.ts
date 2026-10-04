import { resolve } from 'node:path';
import { defineConfig } from 'vite';

// Library build of the demo for the portfolio site (`npm run sync:site` copies web/dist into the site repo).
// Not minified: the site's own build minifies it, and the vendored copy stays readable in diffs.
export default defineConfig({
  build: {
    lib: { entry: resolve(import.meta.dirname, 'src/index.ts'), formats: ['es'], fileName: () => 'maze-bot.js' },
    outDir: resolve(import.meta.dirname, 'dist'),
    emptyOutDir: true,
    minify: false,
    rollupOptions: { output: { chunkFileNames: 'maze-bot-[name]-[hash].js' } },
  },
});
