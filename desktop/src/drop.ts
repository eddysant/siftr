/**
 * Resolving dropped items to absolute paths.
 *
 * Two sources: photos dragged from the grid carry their paths as JSON on the
 * drag event, and photos dragged in from Finder arrive as File objects that only
 * the preload can resolve (`File.path` was removed in Electron 32+).
 */
export const SIFTR_PATHS = 'application/x-siftr-paths';

export function pathsFromDrop(event: React.DragEvent): string[] {
    const internal = event.dataTransfer.getData(SIFTR_PATHS);
    if (internal) {
        try {
            const parsed = JSON.parse(internal);
            return Array.isArray(parsed) ? (parsed as string[]) : [];
        } catch {
            return [];
        }
    }
    return window.api.pathsForFiles(Array.from(event.dataTransfer.files));
}

/**
 * Holding Alt while dropping means "these are counter-examples".
 *
 * Negatives sharpen a tag's threshold, but they need a gesture that cannot be
 * performed by accident — dropping the wrong way round would teach the tag to
 * reject the very thing it should match.
 */
export function isNegativeDrop(event: React.DragEvent): boolean {
    return event.altKey;
}
