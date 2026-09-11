export interface MediaFile {
    id: number;
    path: string;
    name: string;
    kind: 'image' | 'video';
    size: number;
    tags: string[];
    people: string[];
}

export interface Person {
    name: string;
    references: number;
    files: number;
}

export interface FaceCluster {
    size: number;
    files: number;
    face_ids: number[];
    sample_file_id: number;
    sample_path: string | null;
}

export interface Tag {
    name: string;
    threshold: number;
    examples: number;
    matches: number;
    /** Where matches are filed in move mode; null means they stay put. */
    destination: string | null;
}

export type OrganizeMode = 'rename' | 'move' | 'off';

export interface Job {
    id: string;
    kind: string;
    state: 'running' | 'done' | 'failed' | 'cancelled';
    message: string;
    current: number;
    total: number;
    result: unknown;
    error: string | null;
}

export type MatchMode = 'any' | 'all';

declare global {
    interface Window {
        api: {
            request<T>(method: string, path: string, body?: unknown): Promise<T>;
            thumbUrl(path: string, size?: number): string;
            chooseFolder(): Promise<string | null>;
            revealInFinder(path: string): Promise<void>;
            pathsForFiles(files: File[]): string[];
            onServiceError(handler: (message: string) => void): void;
        };
    }
}
