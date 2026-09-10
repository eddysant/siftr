import type { Job, MatchMode, MediaFile, Tag } from './types';

/**
 * Every call goes through main over IPC rather than fetch(), so the bearer token
 * stays in the main process and never reaches the renderer.
 */
const call = <T,>(method: string, path: string, body?: unknown): Promise<T> =>
    window.api.request<T>(method, path, body);

export const getLibrary = () =>
    call<{ files: MediaFile[]; roots: string[] }>('GET', '/api/library');

export const getTags = () => call<{ tags: Tag[] }>('GET', '/api/tags');

export const teachTag = (name: string, paths: string[], replace = false) =>
    call<{ name: string; examples: number; threshold: number; cohesion: number }>(
        'POST',
        '/api/tags',
        { name, paths, replace },
    );

export const forgetTag = (name: string) =>
    call<{ deleted: string }>('DELETE', `/api/tags/${encodeURIComponent(name)}`);

export const setOverride = (path: string, tag: string, state: 'on' | 'off' | null) =>
    call<unknown>('POST', '/api/override', { path, tag, state });

export const startIndex = (folder: string) =>
    call<Job>('POST', '/api/index', { folder, detect_faces: true });

export const startScore = (rename: boolean) =>
    call<Job>('POST', `/api/score?rename=${rename}`);

export const getJob = (id: string) => call<Job>('GET', `/api/jobs/${id}`);

export const undoRenames = () =>
    call<{ restored: number; errors: string[] }>('POST', '/api/renames/undo');

export const search = (tags: string[], mode: MatchMode) =>
    call<{ files: MediaFile[] }>(
        'GET',
        `/api/search?tags=${encodeURIComponent(tags.join(','))}&mode=${mode}`,
    );

/** Poll a job to completion, reporting progress as it goes. */
export async function followJob(id: string, onProgress: (job: Job) => void): Promise<Job> {
    for (;;) {
        const job = await getJob(id);
        onProgress(job);
        if (job.state !== 'running') return job;
        await new Promise((resolve) => setTimeout(resolve, 500));
    }
}
