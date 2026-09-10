import { spawn, type ChildProcess } from 'node:child_process';

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

export function startPythonService(
    port: number,
    pythonBin = 'siftr',
): Promise<ServiceHandle> {
    return new Promise((resolve, reject) => {
        let child: ChildProcess;
        try {
            child = spawn(pythonBin, ['serve', '--port', String(port)], {
                stdio: ['ignore', 'pipe', 'pipe'],
            });
        } catch (error) {
            reject(new Error(`could not start "${pythonBin}": ${String(error)}`));
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
            reject(
                new Error(
                    `could not start "${pythonBin}". Install it with: pip install 'siftr[ui]'\n${String(error)}`,
                ),
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
