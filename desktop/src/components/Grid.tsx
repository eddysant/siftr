import { useEffect, useMemo, useRef, useState } from 'react';
import type { MediaFile } from '../types';

interface Props {
    files: MediaFile[];
    selected: Set<string>;
    onSelect: (path: string, additive: boolean) => void;
    onToggleTag: (file: MediaFile, tag: string) => void;
    onReveal: (path: string) => void;
}

const CELL = 190;
const GAP = 10;
const OVERSCAN = 2;

/**
 * Thumbnail grid with hand-rolled windowing.
 *
 * A library is tens of thousands of files and each cell holds an <img>; mounting
 * them all would exhaust memory long before the user scrolled. Only the visible
 * slice plus a small overscan is rendered, with spacer rows above and below
 * holding the scrollbar at the right height.
 */
export function Grid({ files, selected, onSelect, onToggleTag, onReveal }: Props) {
    const scroller = useRef<HTMLDivElement>(null);
    const [scrollTop, setScrollTop] = useState(0);
    const [viewport, setViewport] = useState({ width: 0, height: 0 });

    useEffect(() => {
        const element = scroller.current;
        if (!element) return;
        // ResizeObserver rather than a window resize listener: the grid also
        // changes width when the rail is toggled, which fires no window event.
        const observer = new ResizeObserver(([entry]) =>
            setViewport({ width: entry.contentRect.width, height: entry.contentRect.height }),
        );
        observer.observe(element);
        return () => observer.disconnect();
    }, []);

    const columns = Math.max(1, Math.floor((viewport.width + GAP) / (CELL + GAP)));
    const rowHeight = CELL + GAP;
    const totalRows = Math.ceil(files.length / columns);

    const { start, end, padTop, padBottom } = useMemo(() => {
        const firstRow = Math.max(0, Math.floor(scrollTop / rowHeight) - OVERSCAN);
        const visibleRows = Math.ceil(viewport.height / rowHeight) + OVERSCAN * 2;
        const lastRow = Math.min(totalRows, firstRow + visibleRows);
        return {
            start: firstRow * columns,
            end: lastRow * columns,
            padTop: firstRow * rowHeight,
            padBottom: Math.max(0, (totalRows - lastRow) * rowHeight),
        };
    }, [scrollTop, viewport.height, rowHeight, totalRows, columns]);

    const slice = files.slice(start, end);

    const handleDragStart = (event: React.DragEvent, file: MediaFile) => {
        // Dragging a file that is part of the selection drags the whole
        // selection; dragging an unselected one drags just that file.
        const paths = selected.has(file.path) ? Array.from(selected) : [file.path];
        event.dataTransfer.setData('application/x-siftr-paths', JSON.stringify(paths));
        event.dataTransfer.effectAllowed = 'copy';
    };

    return (
        <div
            className="grid-scroll"
            ref={scroller}
            onScroll={(event) => setScrollTop(event.currentTarget.scrollTop)}
        >
            {/* The empty state lives INSIDE the scroller rather than replacing
                it. Returning early here left `scroller.current` null on the
                first render — when the library is always still empty — so the
                ResizeObserver was never attached, the effect never re-ran, and
                the grid stayed stuck at a zero-width viewport (one column, a
                handful of cells) forever after the files arrived. */}
            {!files.length && (
                <div className="grid-empty">
                    <p>Nothing to show.</p>
                    <p className="muted">Open a folder and index it, or loosen the tag filter.</p>
                </div>
            )}
            <div
                className="grid"
                style={{ gridTemplateColumns: `repeat(${columns}, ${CELL}px)`, gap: GAP }}
            >
                {padTop > 0 && <div style={{ gridColumn: '1 / -1', height: padTop }} />}

                {slice.map((file) => (
                    <figure
                        key={file.path}
                        className={`cell ${selected.has(file.path) ? 'selected' : ''}`}
                        draggable
                        onDragStart={(event) => handleDragStart(event, file)}
                        onClick={(event) => onSelect(file.path, event.metaKey || event.shiftKey)}
                        onDoubleClick={() => onReveal(file.path)}
                        style={{ height: CELL }}
                    >
                        <img
                            src={window.api.thumbUrl(file.path, 384)}
                            alt={file.name}
                            loading="lazy"
                            draggable={false}
                        />
                        {file.kind === 'video' && <span className="badge-video">▶</span>}
                        <figcaption>
                            <span className="cell-name" title={file.path}>
                                {file.name}
                            </span>
                            <span className="chips">
                                {file.tags.map((tag) => (
                                    <button
                                        key={tag}
                                        type="button"
                                        className="chip"
                                        title={`Remove “${tag}” from this photo`}
                                        onClick={(event) => {
                                            event.stopPropagation();
                                            onToggleTag(file, tag);
                                        }}
                                    >
                                        {tag}
                                    </button>
                                ))}
                                {file.people.map((person) => (
                                    <span key={person} className="chip person">
                                        {person}
                                    </span>
                                ))}
                            </span>
                        </figcaption>
                    </figure>
                ))}

                {padBottom > 0 && <div style={{ gridColumn: '1 / -1', height: padBottom }} />}
            </div>
        </div>
    );
}
