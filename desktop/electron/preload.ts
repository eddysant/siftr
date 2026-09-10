import { contextBridge, ipcRenderer, webUtils } from 'electron';

/**
 * The only surface the renderer gets. `contextIsolation` is on and
 * `nodeIntegration` off, so everything the UI can do is enumerated here.
 *
 * The API token deliberately never crosses this bridge: the renderer asks main
 * to make requests on its behalf, so a compromised renderer cannot exfiltrate a
 * credential it never held.
 */
contextBridge.exposeInMainWorld('api', {
    request: (method: string, path: string, body?: unknown) =>
        ipcRenderer.invoke('siftr:request', { method, path, body }),

    thumbUrl: (path: string, size = 320) =>
        `siftr://thumb/${encodeURIComponent(path)}?w=${size}`,

    chooseFolder: () => ipcRenderer.invoke('dialog:openDirectory'),

    revealInFinder: (path: string) => ipcRenderer.invoke('shell:reveal', path),

    /**
     * Resolve dropped Files to absolute paths. `File.path` was removed in
     * Electron 32+; webUtils is the supported replacement and must be called
     * from the preload, not the renderer.
     */
    pathsForFiles: (files: File[]) => files.map((file) => webUtils.getPathForFile(file)),

    onServiceError: (handler: (message: string) => void) =>
        ipcRenderer.on('siftr:service-error', (_event, message: string) => handler(message)),
});
