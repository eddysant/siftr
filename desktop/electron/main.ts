import { app, BrowserWindow, dialog, ipcMain, net, protocol, shell } from 'electron';
import { existsSync } from 'node:fs';
import path from 'node:path';
import { buildApplicationMenu, configureAboutPanel } from './menu';
import { fileURLToPath } from 'node:url';
import {
    resolveServiceCommand,
    ServiceUnavailable,
    startPythonService,
    type ServiceHandle,
} from './pythonService';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DEV_URL = process.env.VITE_DEV_SERVER_URL;
const SERVICE_PORT = Number(process.env.SIFTR_PORT ?? 8765);

// Exposes the Chrome DevTools Protocol so the UI can be driven and screenshotted
// headlessly in tests, mirroring photo-slap's PHOTO_SLAP_DEBUG_PORT.
const DEBUG_PORT = process.env.SIFTR_DEBUG_PORT;
if (DEBUG_PORT) app.commandLine.appendSwitch('remote-debugging-port', DEBUG_PORT);

let window: BrowserWindow | null = null;
let service: ServiceHandle | null = null;

/**
 * `siftr://` fronts the Python thumbnail endpoint.
 *
 * Two reasons not to point <img> straight at the API: an <img> tag cannot send
 * an Authorization header, and putting the token in a query string would leak it
 * into the DOM and any logs. Main holds the token and proxies instead.
 */
protocol.registerSchemesAsPrivileged([
    {
        scheme: 'siftr',
        privileges: { standard: true, secure: true, supportFetchAPI: true, corsEnabled: true },
    },
]);

function contentSecurityPolicy(): string {
    const base = [
        "default-src 'none'",
        "img-src siftr: data:",
        // React sets inline styles via style={{...}}; there is no way around this one.
        "style-src 'self' 'unsafe-inline'",
        "font-src 'self'",
    ];
    if (DEV_URL) {
        // Vite's dev client needs its inline preamble and a websocket for HMR.
        base.push("script-src 'self' 'unsafe-inline'", `connect-src ${DEV_URL} ws://localhost:* siftr:`);
    } else {
        base.push("script-src 'self'", "connect-src siftr:");
    }
    return base.join('; ');
}

function registerThumbProtocol(handle: ServiceHandle): void {
    protocol.handle('siftr', async (request) => {
        const url = new URL(request.url);
        if (url.hostname !== 'thumb') return new Response('not found', { status: 404 });

        // The leading path segment is the encoded absolute file path. The Python
        // side re-checks it against its own allowlist; this is not the only guard.
        const filePath = decodeURIComponent(url.pathname.replace(/^\//, ''));
        const size = url.searchParams.get('w') ?? '320';
        const target = `${handle.url}/api/thumb?path=${encodeURIComponent(filePath)}&size=${size}`;

        try {
            const response = await net.fetch(target, {
                headers: { Authorization: `Bearer ${handle.token}` },
            });
            return new Response(response.body, {
                status: response.status,
                headers: {
                    'Content-Type': response.headers.get('content-type') ?? 'image/jpeg',
                    'Cache-Control': 'no-cache',
                },
            });
        } catch (error) {
            return new Response(String(error), { status: 502 });
        }
    });
}

function createWindow(): void {
    window = new BrowserWindow({
        width: 1400,
        height: 900,
        minWidth: 900,
        minHeight: 600,
        backgroundColor: '#14131a',
        icon: path.join(__dirname, '..', 'build', 'icon.png'),
        titleBarStyle: 'hiddenInset',
        webPreferences: {
            preload: path.join(__dirname, 'preload.mjs'),
            contextIsolation: true,
            nodeIntegration: false,
        },
    });

    window.webContents.session.webRequest.onHeadersReceived((details, callback) => {
        callback({
            responseHeaders: {
                ...details.responseHeaders,
                'Content-Security-Policy': [contentSecurityPolicy()],
            },
        });
    });

    // A privileged renderer must never navigate to arbitrary content; anything
    // that is not our own page goes to the real browser instead.
    window.webContents.setWindowOpenHandler(({ url }) => {
        shell.openExternal(url);
        return { action: 'deny' };
    });
    window.webContents.on('will-navigate', (event, url) => {
        if (DEV_URL && url.startsWith(DEV_URL)) return;
        event.preventDefault();
        shell.openExternal(url);
    });
    window.webContents.session.setPermissionRequestHandler((_wc, _permission, callback) =>
        callback(false),
    );

    if (DEV_URL) window.loadURL(DEV_URL);
    else window.loadFile(path.join(__dirname, '../dist/index.html'));
}

ipcMain.handle('siftr:request', async (_event, { method, path: apiPath, body }) => {
    if (!service) throw new Error('the siftr service is not running');
    const response = await net.fetch(`${service.url}${apiPath}`, {
        method,
        headers: {
            Authorization: `Bearer ${service.token}`,
            ...(body ? { 'Content-Type': 'application/json' } : {}),
        },
        ...(body ? { body: JSON.stringify(body) } : {}),
    });
    const text = await response.text();
    const parsed = text ? JSON.parse(text) : null;
    if (!response.ok) {
        throw new Error(parsed?.detail ? String(parsed.detail) : `HTTP ${response.status}`);
    }
    return parsed;
});

ipcMain.handle('dialog:openDirectory', async () => {
    const result = await dialog.showOpenDialog({ properties: ['openDirectory'] });
    return result.canceled ? null : result.filePaths[0];
});

ipcMain.handle('shell:reveal', (_event, target: string) => shell.showItemInFolder(target));

app.whenReady().then(async () => {
    configureAboutPanel();
    buildApplicationMenu();

    // In development the app runs under the Electron binary, whose bundle owns
    // the Dock icon; setting it explicitly makes `npm run dev` show siftr's icon
    // instead of Electron's. A packaged build gets it from the bundle already.
    if (DEV_URL && process.platform === 'darwin') {
        const devIcon = path.join(__dirname, '..', 'build', 'icon.png');
        if (existsSync(devIcon)) app.dock?.setIcon(devIcon);
    }

    try {
        service = await startPythonService(SERVICE_PORT, resolveServiceCommand(process.resourcesPath));
        registerThumbProtocol(service);
    } catch (error) {
        // Start the window regardless so the failure is visible in the UI rather
        // than as a silent no-launch.
        createWindow();
        const message =
            error instanceof ServiceUnavailable ? error.guidance : String(error);
        window?.webContents.once('did-finish-load', () =>
            window?.webContents.send('siftr:service-error', message),
        );
        return;
    }
    createWindow();
});

app.on('window-all-closed', () => {
    service?.stop();
    if (process.platform !== 'darwin') app.quit();
});

app.on('before-quit', () => service?.stop());
