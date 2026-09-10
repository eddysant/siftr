import type { MatchMode, MediaFile } from './types';

export interface FilterState {
    query: string;
    selectedTags: string[];
    mode: MatchMode;
    untaggedOnly: boolean;
}

/**
 * Filter the library client-side.
 *
 * Kept pure and separate from the component so the ALL/ANY semantics — the part
 * most likely to be subtly wrong — can be tested without rendering anything.
 */
export function filterFiles(files: MediaFile[], filters: FilterState): MediaFile[] {
    const query = filters.query.trim().toLowerCase();
    const selected = filters.selectedTags;

    return files.filter((file) => {
        if (query && !file.name.toLowerCase().includes(query)) return false;
        if (filters.untaggedOnly && file.tags.length > 0) return false;

        // No tag selected means "no tag constraint", not "match nothing" —
        // otherwise opening the app would show an empty grid.
        if (selected.length === 0) return true;

        return filters.mode === 'all'
            ? selected.every((tag) => file.tags.includes(tag))
            : selected.some((tag) => file.tags.includes(tag));
    });
}

/** How many files each tag currently covers, for the counts in the rail. */
export function tagCounts(files: MediaFile[]): Record<string, number> {
    const counts: Record<string, number> = {};
    for (const file of files) {
        for (const tag of file.tags) counts[tag] = (counts[tag] ?? 0) + 1;
    }
    return counts;
}

/** Toggle one tag in the selection. */
export function toggleTag(selected: string[], tag: string): string[] {
    return selected.includes(tag) ? selected.filter((t) => t !== tag) : [...selected, tag];
}
