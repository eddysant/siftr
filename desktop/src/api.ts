import type { FaceCluster, Job, MatchMode, MediaFile, Person, Tag } from './types';

/**
 * Every call goes through main over IPC rather than fetch(), so the bearer token
 * stays in the main process and never reaches the renderer.
 */
const call = <T,>(method: string, path: string, body?: unknown): Promise<T> =>
    window.api.request<T>(method, path, body);

export const getLibrary = () =>
    call<{ files: MediaFile[]; roots: string[] }>('GET', '/api/library');

export const getTags = () => call<{ tags: Tag[] }>('GET', '/api/tags');

export const teachTag = (
    name: string,
    paths: string[],
    negatives: string[] = [],
    replace = false,
) =>
    call<{ name: string; examples: number; threshold: number; cohesion: number }>(
        'POST',
        '/api/tags',
        { name, paths, negatives, replace },
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

export const getPeople = () => call<{ people: Person[] }>('GET', '/api/people');

export const addPerson = (name: string, paths: string[]) =>
    call<{ name: string; references: number; skipped: string[] }>('POST', '/api/people', {
        name,
        paths,
    });

export const forgetPerson = (name: string) =>
    call<{ deleted: string }>('DELETE', `/api/people/${encodeURIComponent(name)}`);

export const personFiles = (name: string) =>
    call<{ files: { path: string; name: string; kind: string; score: number }[] }>(
        'GET',
        `/api/people/${encodeURIComponent(name)}/files`,
    );

export const getClusters = () => call<{ clusters: FaceCluster[] }>('GET', '/api/faces/clusters');

export const nameCluster = (name: string, faceIds: number[]) =>
    call<{ name: string; references: number }>('POST', '/api/faces/clusters/name', {
        name,
        face_ids: faceIds,
    });

export const pendingRenames = () =>
    call<{ pending: number; changes: string[][] }>('GET', '/api/renames/pending');

export const cancelJob = (id: string) => call<Job>('POST', `/api/jobs/${id}/cancel`);

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
