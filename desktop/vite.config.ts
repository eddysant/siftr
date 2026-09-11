import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import electron from 'vite-plugin-electron/simple';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

// `__dirname` is unavailable under Vite's native config loader, which is due to
// become the default; deriving it from import.meta keeps this working in both.
const here = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
    plugins: [
        react(),
        // The `simple` form, not the generic entry array. It has a dedicated
        // preload path that emits CommonJS even when the package is ESM — which
        // matters because a sandboxed Electron preload MUST be CJS. Building it
        // as ESM fails with only "Unable to load preload script" in the console,
        // leaving window.api undefined and the renderer dead on first access.
        electron({
            main: { entry: path.join(here, 'electron/main.ts') },
            preload: { input: path.join(here, 'electron/preload.ts') },
        }),
    ],
});
