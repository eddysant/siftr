import { describe, expect, it, vi } from 'vitest';
import { SIFTR_PATHS, isNegativeDrop, pathsFromDrop } from '../drop';

const dragEvent = (data: Record<string, string>, files: File[] = [], altKey = false) =>
    ({
        altKey,
        dataTransfer: {
            getData: (type: string) => data[type] ?? '',
            files,
        },
    }) as unknown as React.DragEvent;

describe('pathsFromDrop', () => {
    it('reads paths dragged from the grid', () => {
        const event = dragEvent({ [SIFTR_PATHS]: JSON.stringify(['/a.jpg', '/b.jpg']) });
        expect(pathsFromDrop(event)).toEqual(['/a.jpg', '/b.jpg']);
    });

    it('falls back to the preload resolver for Finder drops', () => {
        const file = new File([], 'x.jpg');
        vi.stubGlobal('window', { api: { pathsForFiles: () => ['/from/finder.jpg'] } });
        expect(pathsFromDrop(dragEvent({}, [file]))).toEqual(['/from/finder.jpg']);
        vi.unstubAllGlobals();
    });

    it('returns nothing for malformed internal data rather than throwing', () => {
        expect(pathsFromDrop(dragEvent({ [SIFTR_PATHS]: '{not json' }))).toEqual([]);
    });

    it('returns nothing when the payload is not an array', () => {
        expect(pathsFromDrop(dragEvent({ [SIFTR_PATHS]: '"a string"' }))).toEqual([]);
    });
});

describe('isNegativeDrop', () => {
    it('is true only with Alt held', () => {
        expect(isNegativeDrop(dragEvent({}, [], true))).toBe(true);
        expect(isNegativeDrop(dragEvent({}, [], false))).toBe(false);
    });
});
