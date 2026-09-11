import type {
    BoundaryFile,
    DuplicateGroup,
    FaceCluster,
    Job,
    MatchMode,
    MediaFile,
    OrganizeMode,
    Person,
    Tag,
} from './types';

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

export const getBoundary = (tag: string, limit = 12) =>
    call<{ tag: string; threshold: number; files: BoundaryFile[] }>(
        'GET',
        `/api/tags/${encodeURIComponent(tag)}/boundary?limit=${limit}`,
    );

export const reviewBoundary = (tag: string, path: string, isMatch: boolean) =>
    call<{ tag: string; threshold: number; examples: number; rejections: number }>(
        'POST',
        `/api/tags/${encodeURIComponent(tag)}/review`,
        { path, is_match: isMatch },
    );

export const collectDuplicates = (
    destination: string,
    distance: number,
    mode: 'symlink' | 'copy' | 'move',
) =>
    call<{ groups: number; collected: number; skipped: number; errors: string[] }>(
        'POST',
        '/api/duplicates/collect',
        { destination, distance, mode },
    );

export const verifyTag = (tag: string, phrase: string, candidates = 200) =>
    call<Job>(
        'POST',
        `/api/tags/${encodeURIComponent(tag)}/verify?phrase=${encodeURIComponent(phrase)}` +
            `&candidates=${candidates}`,
    );

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

export const getSettings = () => call<{ organize_mode: OrganizeMode }>('GET', '/api/settings');

export const setOrganizeMode = (organize_mode: OrganizeMode) =>
    call<{ organize_mode: OrganizeMode }>('PUT', '/api/settings', { organize_mode });

export const setDestination = (
    tag: string,
    destination: string | null,
    inverse = false,
) =>
    call<{ name: string; destination: string | null; inverse: boolean }>(
        'PUT',
        `/api/tags/${encodeURIComponent(tag)}/destination`,
        { destination, inverse },
    );

export const getDuplicates = (distance?: number) =>
    call<{ distance: number; groups: DuplicateGroup[]; redundant_total: number }>(
        'GET',
        `/api/duplicates${distance === undefined ? '' : `?distance=${distance}`}`,
    );

export const organizeNow = (dryRun = false) =>
    call<{
        mode: OrganizeMode;
        changed: number;
        skipped: number;
        paired?: number;
        errors: string[];
    }>('POST', `/api/organize?dry_run=${dryRun}`);

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
