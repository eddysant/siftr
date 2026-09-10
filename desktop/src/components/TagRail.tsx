import { useState } from 'react';
import type { MatchMode, Tag } from '../types';

interface Props {
    tags: Tag[];
    counts: Record<string, number>;
    selected: string[];
    mode: MatchMode;
    busy: boolean;
    onToggle: (name: string) => void;
    onModeChange: (mode: MatchMode) => void;
    onDropFiles: (tagName: string, paths: string[]) => void;
    onCreateTag: (name: string, paths: string[]) => void;
    onForget: (name: string) => void;
}

/**
 * The left rail: every tag is a drop target.
 *
 * Dropping photos onto a tag is how a tag is taught, so the drop affordance has
 * to be unmistakable — the row lights up and says what will happen, because the
 * gesture has a real consequence (it retrains the tag and can rename files).
 */
export function TagRail({
    tags,
    counts,
    selected,
    mode,
    busy,
    onToggle,
    onModeChange,
    onDropFiles,
    onCreateTag,
    onForget,
}: Props) {
    const [dropTarget, setDropTarget] = useState<string | null>(null);
    const [newTagOpen, setNewTagOpen] = useState(false);
    const [newTagName, setNewTagName] = useState('');

    const pathsFromDrop = (event: React.DragEvent): string[] => {
        // Photos dragged from the grid carry their paths as JSON; photos dragged
        // in from Finder arrive as File objects that only the preload can resolve.
        const internal = event.dataTransfer.getData('application/x-siftr-paths');
        if (internal) {
            try {
                return JSON.parse(internal) as string[];
            } catch {
                return [];
            }
        }
        return window.api.pathsForFiles(Array.from(event.dataTransfer.files));
    };

    const handleDrop = (event: React.DragEvent, tagName: string) => {
        event.preventDefault();
        setDropTarget(null);
        const paths = pathsFromDrop(event);
        if (paths.length) onDropFiles(tagName, paths);
    };

    return (
        <aside className="rail">
            <div className="rail-head">
                <h2>Tags</h2>
                <div className="mode-toggle" role="group" aria-label="Match mode">
                    {(['any', 'all'] as MatchMode[]).map((m) => (
                        <button
                            key={m}
                            type="button"
                            className={mode === m ? 'active' : ''}
                            onClick={() => onModeChange(m)}
                            title={
                                m === 'any'
                                    ? 'Show photos matching any selected tag'
                                    : 'Show only photos matching every selected tag'
                            }
                        >
                            {m.toUpperCase()}
                        </button>
                    ))}
                </div>
            </div>

            <ul className="tag-list">
                {tags.map((tag) => {
                    const isSelected = selected.includes(tag.name);
                    return (
                        <li
                            key={tag.name}
                            className={[
                                'tag-row',
                                isSelected ? 'selected' : '',
                                dropTarget === tag.name ? 'drop-active' : '',
                            ].join(' ')}
                            onDragOver={(event) => {
                                event.preventDefault();
                                setDropTarget(tag.name);
                            }}
                            onDragLeave={() => setDropTarget((t) => (t === tag.name ? null : t))}
                            onDrop={(event) => handleDrop(event, tag.name)}
                        >
                            <button
                                type="button"
                                className="tag-main"
                                onClick={() => onToggle(tag.name)}
                                disabled={busy}
                            >
                                <span className="tag-name">{tag.name}</span>
                                <span className="tag-count">{counts[tag.name] ?? 0}</span>
                            </button>
                            <div className="tag-meta">
                                <span title="Training examples">{tag.examples} ex</span>
                                <span title="Match threshold">{tag.threshold.toFixed(2)}</span>
                                <button
                                    type="button"
                                    className="tag-forget"
                                    onClick={() => onForget(tag.name)}
                                    disabled={busy}
                                    title={`Forget "${tag.name}"`}
                                >
                                    ×
                                </button>
                            </div>
                            {dropTarget === tag.name && (
                                <div className="drop-hint">teach “{tag.name}”</div>
                            )}
                        </li>
                    );
                })}
            </ul>

            <div
                className={`new-tag ${dropTarget === '__new__' ? 'drop-active' : ''}`}
                onDragOver={(event) => {
                    event.preventDefault();
                    setDropTarget('__new__');
                }}
                onDragLeave={() => setDropTarget(null)}
                onDrop={(event) => {
                    event.preventDefault();
                    setDropTarget(null);
                    const paths = pathsFromDrop(event);
                    if (!paths.length) return;
                    const name = window.prompt('Name this tag');
                    if (name?.trim()) onCreateTag(name.trim(), paths);
                }}
            >
                {newTagOpen ? (
                    <form
                        onSubmit={(event) => {
                            event.preventDefault();
                            if (newTagName.trim()) {
                                onCreateTag(newTagName.trim(), []);
                                setNewTagName('');
                                setNewTagOpen(false);
                            }
                        }}
                    >
                        <input
                            autoFocus
                            value={newTagName}
                            placeholder="tag name"
                            onChange={(event) => setNewTagName(event.target.value)}
                            onBlur={() => setNewTagOpen(false)}
                        />
                    </form>
                ) : (
                    <button type="button" onClick={() => setNewTagOpen(true)} disabled={busy}>
                        + drop photos here to make a tag
                    </button>
                )}
            </div>
        </aside>
    );
}
