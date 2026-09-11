import { app, BrowserWindow, Menu, shell, type MenuItemConstructorOptions } from 'electron';

/**
 * The application menu.
 *
 * Electron's default menu is generic and advertises the framework rather than
 * the app, so the whole thing is replaced. Items that act on the library
 * dispatch over `menu:action` to the renderer, which already owns that state —
 * duplicating the logic in main would mean two sources of truth for what
 * "Re-score" means.
 *
 * Accelerators are registered normally here (unlike photo-slap, which displays
 * them without registering): siftr has no single-key renderer shortcuts that a
 * registered accelerator could swallow from a text field.
 */

export type MenuAction =
    | 'open-folder'
    | 'rescore'
    | 'organize'
    | 'undo-changes'
    | 'reveal'
    | 'mode-rename'
    | 'mode-move'
    | 'mode-off'
    | 'clear-selection';

const REPO = 'https://github.com/eddysant/siftr';

function send(action: MenuAction): void {
    BrowserWindow.getFocusedWindow()?.webContents.send('menu:action', action);
}

/**
 * The template as plain data, separated from `Menu.setApplicationMenu` so its
 * shape can be asserted in tests without an Electron runtime. A menu that
 * silently loses an item is otherwise only discoverable by opening it.
 */
export function menuTemplate(
    dispatch: (action: MenuAction) => void = send,
    openExternal: (url: string) => void = (url) => void shell.openExternal(url),
    appName = 'siftr',
    version = '0.0.0',
): MenuItemConstructorOptions[] {
    void version;
    return [
        {
            label: appName,
            submenu: [
                { role: 'about' },
                { type: 'separator' },
                {
                    label: 'Organize Matches',
                    submenu: [
                        {
                            label: 'Rename with [tag]',
                            click: () => dispatch('mode-rename'),
                        },
                        { label: 'Move to Tag Folder', click: () => dispatch('mode-move') },
                        { label: 'Leave Files Alone', click: () => dispatch('mode-off') },
                    ],
                },
                { type: 'separator' },
                { role: 'services' },
                { type: 'separator' },
                { role: 'hide' },
                { role: 'hideOthers' },
                { role: 'unhide' },
                { type: 'separator' },
                { role: 'quit' },
            ],
        },
        {
            label: 'File',
            submenu: [
                {
                    label: 'Open Folder…',
                    accelerator: 'CmdOrCtrl+O',
                    click: () => dispatch('open-folder'),
                },
                { type: 'separator' },
                {
                    label: 'Re-score Library',
                    accelerator: 'CmdOrCtrl+R',
                    click: () => dispatch('rescore'),
                },
                {
                    label: 'Apply Organize Policy',
                    accelerator: 'CmdOrCtrl+Shift+O',
                    click: () => dispatch('organize'),
                },
                { type: 'separator' },
                {
                    label: 'Undo Last File Change',
                    accelerator: 'CmdOrCtrl+Z',
                    click: () => dispatch('undo-changes'),
                },
                {
                    label: 'Reveal Selection in Finder',
                    accelerator: 'CmdOrCtrl+Shift+R',
                    click: () => dispatch('reveal'),
                },
            ],
        },
        {
            label: 'Edit',
            // Real roles, not custom handlers: the search field and the name
            // prompt need working copy/paste and native undo.
            submenu: [
                { role: 'undo' },
                { role: 'redo' },
                { type: 'separator' },
                { role: 'cut' },
                { role: 'copy' },
                { role: 'paste' },
                { role: 'selectAll' },
                { type: 'separator' },
                {
                    label: 'Deselect Photos',
                    accelerator: 'Escape',
                    click: () => dispatch('clear-selection'),
                },
            ],
        },
        {
            label: 'View',
            submenu: [
                { role: 'reload' },
                { role: 'toggleDevTools' },
                { type: 'separator' },
                { role: 'resetZoom' },
                { role: 'zoomIn' },
                { role: 'zoomOut' },
                { type: 'separator' },
                { role: 'togglefullscreen' },
            ],
        },
        { label: 'Window', submenu: [{ role: 'minimize' }, { role: 'zoom' }, { role: 'close' }] },
        {
            role: 'help',
            submenu: [
                { label: 'siftr on GitHub', click: () => openExternal(REPO) },
                {
                    label: 'Report an Issue',
                    click: () => openExternal(`${REPO}/issues/new`),
                },
            ],
        },
    ];
}

export function buildApplicationMenu(): void {
    Menu.setApplicationMenu(
        Menu.buildFromTemplate(menuTemplate(send, (url) => void shell.openExternal(url), app.name)),
    );
}

export function configureAboutPanel(): void {
    app.setAboutPanelOptions({
        applicationName: 'siftr',
        applicationVersion: app.getVersion(),
        copyright: 'Find media by example.',
        credits: 'CLIP embeddings via open_clip · faces via InsightFace',
    });
}
