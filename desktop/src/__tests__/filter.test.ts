import { describe, expect, it } from 'vitest';
import { filterFiles, tagCounts, toggleTag } from '../filter';
import type { MediaFile } from '../types';

const file = (name: string, tags: string[]): MediaFile => ({
    id: name.length,
    path: `/lib/${name}`,
    name,
    kind: 'image',
    size: 1,
    tags,
    people: [],
});

const library = [
    file('a.jpg', ['glaze', 'outdoor']),
    file('b.jpg', ['glaze']),
    file('c.jpg', ['outdoor']),
    file('d.jpg', []),
];

const base = { query: '', selectedTags: [], mode: 'any' as const, untaggedOnly: false };

describe('filterFiles', () => {
    it('returns everything when nothing is selected', () => {
        expect(filterFiles(library, base)).toHaveLength(4);
    });

    it('ANY is the union', () => {
        const result = filterFiles(library, { ...base, selectedTags: ['glaze', 'outdoor'] });
        expect(result.map((f) => f.name)).toEqual(['a.jpg', 'b.jpg', 'c.jpg']);
    });

    it('ALL is the intersection', () => {
        const result = filterFiles(library, {
            ...base,
            selectedTags: ['glaze', 'outdoor'],
            mode: 'all',
        });
        expect(result.map((f) => f.name)).toEqual(['a.jpg']);
    });

    it('ALL with a single tag matches that tag', () => {
        const result = filterFiles(library, { ...base, selectedTags: ['glaze'], mode: 'all' });
        expect(result.map((f) => f.name)).toEqual(['a.jpg', 'b.jpg']);
    });

    it('ALL with an unmatched tag yields nothing', () => {
        const result = filterFiles(library, {
            ...base,
            selectedTags: ['glaze', 'nonexistent'],
            mode: 'all',
        });
        expect(result).toEqual([]);
    });

    it('filters by filename query', () => {
        expect(filterFiles(library, { ...base, query: 'a.' }).map((f) => f.name)).toEqual(['a.jpg']);
    });

    it('combines query with tag selection', () => {
        const result = filterFiles(library, {
            ...base,
            query: 'b',
            selectedTags: ['glaze'],
        });
        expect(result.map((f) => f.name)).toEqual(['b.jpg']);
    });

    it('untaggedOnly shows files no tag matched', () => {
        const result = filterFiles(library, { ...base, untaggedOnly: true });
        expect(result.map((f) => f.name)).toEqual(['d.jpg']);
    });
});

describe('tagCounts', () => {
    it('counts each tag across the library', () => {
        expect(tagCounts(library)).toEqual({ glaze: 2, outdoor: 2 });
    });

    it('is empty for an untagged library', () => {
        expect(tagCounts([file('x.jpg', [])])).toEqual({});
    });
});

describe('toggleTag', () => {
    it('adds a tag that is not selected', () => {
        expect(toggleTag(['a'], 'b')).toEqual(['a', 'b']);
    });

    it('removes a tag that is selected', () => {
        expect(toggleTag(['a', 'b'], 'a')).toEqual(['b']);
    });
});
