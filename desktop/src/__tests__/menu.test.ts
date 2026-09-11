import { describe, expect, it, vi } from 'vitest';

// The module imports electron for Menu/app/shell; only the template shape is
// under test, so a stub is enough to let it load in Node.
vi.mock('electron', () => ({
    app: { name: 'siftr', getVersion: () => '0.1.0', setAboutPanelOptions: vi.fn() },
    BrowserWindow: { getFocusedWindow: () => null },
    Menu: { buildFromTemplate: vi.fn(), setApplicationMenu: vi.fn() },
    shell: { openExternal: vi.fn() },
}));

const { menuTemplate } = await import('../../electron/menu');

type Item = { label?: string; role?: string; type?: string; click?: () => void; submenu?: Item[] };

const template = () => menuTemplate(vi.fn(), vi.fn(), 'siftr') as Item[];

const labels = (items: Item[] = []) =>
    items.filter((i) => i.type !== 'separator').map((i) => i.label ?? i.role);

const menu = (name: string) => {
    const found = template().find((m) => (m.label ?? m.role) === name);
    if (!found) throw new Error(`no "${name}" menu`);
    return found.submenu ?? [];
};

describe('application menu', () => {
    it('is not the default Electron menu', () => {
        const top = labels(template());
        expect(top[0]).toBe('siftr');
        expect(top).not.toContain('Electron');
    });

    it('has the standard macOS top-level structure', () => {
        expect(labels(template())).toEqual(['siftr', 'File', 'Edit', 'View', 'Window', 'help']);
    });

    it('the app menu offers About and Quit', () => {
        expect(labels(menu('siftr'))).toContain('about');
        expect(labels(menu('siftr'))).toContain('quit');
    });

    it('exposes the three organize modes', () => {
        const organize = menu('siftr').find((i) => i.label === 'Organize Matches');
        expect(labels(organize?.submenu)).toEqual([
            'Rename with [tag]',
            'Move to Tag Folder',
            'Leave Files Alone',
        ]);
    });

    it('File carries the library actions', () => {
        expect(labels(menu('File'))).toEqual([
            'Open Folder…',
            'Re-score Library',
            'Apply Organize Policy',
            'Undo Last File Change',
            'Reveal Selection in Finder',
        ]);
    });

    it('Edit keeps the native roles that text fields need', () => {
        const roles = labels(menu('Edit'));
        for (const role of ['undo', 'redo', 'cut', 'copy', 'paste', 'selectAll']) {
            expect(roles).toContain(role);
        }
    });

    it('every custom item dispatches an action', () => {
        const dispatch = vi.fn();
        const built = menuTemplate(dispatch, vi.fn(), 'siftr') as Item[];
        const clickable: Item[] = [];
        const walk = (items: Item[]) => {
            for (const item of items) {
                if (item.click) clickable.push(item);
                if (item.submenu) walk(item.submenu);
            }
        };
        walk(built);
        // Two Help items open URLs instead; everything else dispatches.
        const appActions = clickable.filter((i) => !i.label?.match(/GitHub|Report an Issue/));
        expect(appActions.length).toBeGreaterThan(0);
        for (const item of appActions) item.click?.();
        expect(dispatch).toHaveBeenCalledTimes(appActions.length);
    });

    it('Help links out rather than dispatching', () => {
        const openExternal = vi.fn();
        const built = menuTemplate(vi.fn(), openExternal, 'siftr') as Item[];
        const help = built.find((m) => m.role === 'help')?.submenu ?? [];
        for (const item of help) item.click?.();
        expect(openExternal).toHaveBeenCalledTimes(2);
        expect(openExternal.mock.calls[0][0]).toContain('github.com/eddysant/siftr');
    });

    it('accelerators do not collide', () => {
        const seen = new Set<string>();
        const walk = (items: Item[]) => {
            for (const item of items) {
                const accel = (item as { accelerator?: string }).accelerator;
                if (accel) {
                    expect(seen.has(accel), `duplicate accelerator ${accel}`).toBe(false);
                    seen.add(accel);
                }
                if (item.submenu) walk(item.submenu);
            }
        };
        walk(template());
    });
});
