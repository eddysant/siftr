import { spawn, type ChildProcess } from 'node:child_process';
import { existsSync } from 'node:fs';
import path from 'node:path';

/**
 * Owns the `siftr serve` child process.
 *
 * The UI cannot shell out to the CLI per action — loading CLIP costs several
 * seconds of torch import plus model init, so a resident process pays that once.
 * The server mints a bearer token at startup and prints it on stdout; that line
 * is the handshake, which keeps the token out of argv (visible in `ps`) and off
 * disk.
 */
export interface ServiceHandle {
    url: string;
    token: string;
    stop: () => void;
}

const READY_TIMEOUT_MS = 60_000;

/**
 * Where to look for the siftr service, in order.
 *
 * A bundled interpreter wins when one is shipped in the app's resources; a
 * packaged build that vendors Python drops it there. Otherwise we fall back to
 * whatever `siftr` is on PATH, which is what a `pip install -e '.[ui]'`
 * development setup provides.
 *
 * PATH is looked up explicitly rather than relying on the inherited environment:
 * a GUI app launched from Finder gets a minimal PATH that does not include
 * Homebrew, pyenv, or a virtualenv's bin, so `siftr` would appear missing on a
 * machine where it is plainly installed.
 */
const EXTRA_PATHS = [
    '/opt/homebrew/bin',
    '/usr/local/bin',
    `${process.env.HOME}/.local/bin`,
    `${process.env.HOME}/Library/Python/3.12/bin`,
];

export function resolveServiceCommand(resourcesPath?: string): string {
    // An explicit override wins over everything. This is the escape hatch for a
    // virtualenv install, which is the common case and is never on the system
    // PATH a GUI app inherits.
    const override = process.env.SIFTR_BIN;
    if (override && existsSync(override)) return override;

    if (resourcesPath) {
        const bundled = path.join(resourcesPath, 'python', 'bin', 'siftr');
        if (existsSync(bundled)) return bundled;
    }
    for (const dir of EXTRA_PATHS) {
        const candidate = path.join(dir, 'siftr');
        if (existsSync(candidate)) return candidate;
    }
    return 'siftr';
}

export function augmentedPath(): string {
    const current = process.env.PATH ?? '';
    const missing = EXTRA_PATHS.filter((dir) => !current.split(':').includes(dir));
    return [current, ...missing].filter(Boolean).join(':');
}

export class ServiceUnavailable extends Error {
    readonly guidance: string;

    constructor(message: string, guidance: string) {
        super(message);
        this.name = 'ServiceUnavailable';
        this.guidance = guidance;
    }
}

const INSTALL_HINT =
    "siftr's Python service was not found.\n\n" +
    "Install it:\n    pip install 'siftr[ui,faces]'\n\n" +
    'Already installed in a virtualenv? A packaged app does not inherit your\n' +
    'shell PATH, so point it at the executable directly:\n' +
    '    SIFTR_BIN=/path/to/.venv/bin/siftr open -a siftr\n\n' +
    'Then reopen this app.';

export function startPythonService(
    port: number,
    pythonBin = 'siftr',
): Promise<ServiceHandle> {
    return new Promise((resolve, reject) => {
        let child: ChildProcess;
        try {
            child = spawn(pythonBin, ['serve', '--port', String(port)], {
                stdio: ['ignore', 'pipe', 'pipe'],
                env: { ...process.env, PATH: augmentedPath() },
            });
        } catch (error) {
            reject(new ServiceUnavailable(String(error), INSTALL_HINT));
            return;
        }

        let token = '';
        let url = '';
        let settled = false;
        let stderrTail = '';

        const timer = setTimeout(() => {
            if (settled) return;
            settled = true;
            child.kill();
            reject(new Error(`siftr serve did not start within ${READY_TIMEOUT_MS / 1000}s`));
        }, READY_TIMEOUT_MS);

        const finish = () => {
            if (settled || !token || !url) return;
            settled = true;
            clearTimeout(timer);
            resolve({ url, token, stop: () => child.kill() });
        };

        child.stdout?.on('data', (chunk: Buffer) => {
            for (const line of chunk.toString().split('\n')) {
                if (line.startsWith('SIFTR_TOKEN=')) token = line.slice('SIFTR_TOKEN='.length).trim();
                if (line.startsWith('SIFTR_URL=')) url = line.slice('SIFTR_URL='.length).trim();
            }
            finish();
        });

        child.stderr?.on('data', (chunk: Buffer) => {
            // Keep only the tail: a traceback is what matters, and uvicorn is chatty.
            stderrTail = (stderrTail + chunk.toString()).slice(-2000);
        });

        child.on('error', (error) => {
            if (settled) return;
            settled = true;
            clearTimeout(timer);
            const isMissing = (error as NodeJS.ErrnoException).code === 'ENOENT';
            reject(
                isMissing
                    ? new ServiceUnavailable(`"${pythonBin}" not found`, INSTALL_HINT)
                    : new Error(`could not start "${pythonBin}": ${String(error)}`),
            );
        });

        child.on('exit', (code) => {
            if (settled) return;
            settled = true;
            clearTimeout(timer);
            reject(new Error(`siftr serve exited with code ${code}\n${stderrTail}`));
        });
    });
}
