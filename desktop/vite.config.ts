import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import electron from 'vite-plugin-electron/simple';
import path from 'node:path';

export default defineConfig({
    plugins: [
        react(),
        // The `simple` form, not the generic entry array. It has a dedicated
        // preload path that emits CommonJS even when the package is ESM — which
        // matters because a sandboxed Electron preload MUST be CJS. Building it
        // as ESM fails with only "Unable to load preload script" in the console,
        // leaving window.api undefined and the renderer dead on first access.
        electron({
            main: { entry: path.join(__dirname, 'electron/main.ts') },
            preload: { input: path.join(__dirname, 'electron/preload.ts') },
        }),
    ],
});
