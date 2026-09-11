import { useState } from 'react';
import { isNegativeDrop, pathsFromDrop } from '../drop';
import type { FaceCluster, MatchMode, OrganizeMode, Person, Tag } from '../types';

interface Props {
    tags: Tag[];
    people: Person[];
    clusters: FaceCluster[];
    counts: Record<string, number>;
    selectedTags: string[];
    selectedPerson: string | null;
    mode: MatchMode;
    busy: boolean;
    onToggleTag: (name: string) => void;
    onSelectPerson: (name: string | null) => void;
    onModeChange: (mode: MatchMode) => void;
    onDropOnTag: (tagName: string, paths: string[], negative: boolean) => void;
    onDropNewTag: (paths: string[]) => void;
    onDropNewPerson: (paths: string[]) => void;
    organizeMode: OrganizeMode;
    onSetDestination: (tag: string) => void;
    onForgetTag: (name: string) => void;
    onForgetPerson: (name: string) => void;
    onNameCluster: (cluster: FaceCluster) => void;
}

/**
 * The left rail: tags and people, both of which are drop targets.
 *
 * Dropping is the primary gesture in this app, so every target lights up and
 * says what it will do — a drop retrains something and can rename files.
 */
export function TagRail(props: Props) {
    const {
        tags,
        people,
        clusters,
        counts,
        selectedTags,
        selectedPerson,
        mode,
        busy,
        onToggleTag,
        onSelectPerson,
        onModeChange,
        onDropOnTag,
        onDropNewTag,
        onDropNewPerson,
        organizeMode,
        onSetDestination,
        onForgetTag,
        onForgetPerson,
        onNameCluster,
    } = props;

    const [dropTarget, setDropTarget] = useState<string | null>(null);
    const [negativeDrop, setNegativeDrop] = useState(false);

    const dragOver = (event: React.DragEvent, key: string) => {
        event.preventDefault();
        setDropTarget(key);
        setNegativeDrop(isNegativeDrop(event));
    };

    const clearDrop = () => {
        setDropTarget(null);
        setNegativeDrop(false);
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
                                    ? 'Photos matching any selected tag'
                                    : 'Only photos matching every selected tag'
                            }
                        >
                            {m.toUpperCase()}
                        </button>
                    ))}
                </div>
            </div>

            <ul className="tag-list">
                {tags.map((tag) => {
                    const key = `tag:${tag.name}`;
                    const active = dropTarget === key;
                    return (
                        <li
                            key={tag.name}
                            className={[
                                'tag-row',
                                selectedTags.includes(tag.name) ? 'selected' : '',
                                active ? 'drop-active' : '',
                                active && negativeDrop ? 'drop-negative' : '',
                            ].join(' ')}
                            onDragOver={(event) => dragOver(event, key)}
                            onDragLeave={clearDrop}
                            onDrop={(event) => {
                                event.preventDefault();
                                const negative = isNegativeDrop(event);
                                const paths = pathsFromDrop(event);
                                clearDrop();
                                if (paths.length) onDropOnTag(tag.name, paths, negative);
                            }}
                        >
                            <button
                                type="button"
                                className="tag-main"
                                onClick={() => onToggleTag(tag.name)}
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
                                    onClick={() => onForgetTag(tag.name)}
                                    disabled={busy}
                                    title={`Forget "${tag.name}"`}
                                >
                                    ×
                                </button>
                            </div>
                            {organizeMode === 'move' && (
                                <button
                                    type="button"
                                    className={`tag-dest ${tag.destination ? '' : 'unset'}`}
                                    onClick={() => onSetDestination(tag.name)}
                                    disabled={busy}
                                    title={
                                        tag.destination
                                            ? `Matches are filed in ${tag.destination}`
                                            : 'Choose where matches are filed'
                                    }
                                >
                                    {tag.destination
                                        ? `→ ${tag.destination.split('/').pop()}`
                                        : '→ choose folder…'}
                                </button>
                            )}
                            {active && (
                                <div className="drop-hint">
                                    {negativeDrop ? `NOT “${tag.name}”` : `teach “${tag.name}”`}
                                </div>
                            )}
                        </li>
                    );
                })}
            </ul>

            <DropZone
                label="+ drop photos here to make a tag"
                active={dropTarget === 'new-tag'}
                onDragOver={(event) => dragOver(event, 'new-tag')}
                onDragLeave={clearDrop}
                onDrop={(event) => {
                    event.preventDefault();
                    const paths = pathsFromDrop(event);
                    clearDrop();
                    if (paths.length) onDropNewTag(paths);
                }}
            />

            <p className="rail-tip">hold ⌥ while dropping to teach a counter-example</p>

            <div className="rail-head">
                <h2>People</h2>
                {selectedPerson && (
                    <button type="button" className="clear" onClick={() => onSelectPerson(null)}>
                        clear
                    </button>
                )}
            </div>

            <ul className="tag-list">
                {people.map((person) => (
                    <li
                        key={person.name}
                        className={`tag-row ${selectedPerson === person.name ? 'selected' : ''} ${
                            dropTarget === `person:${person.name}` ? 'drop-active' : ''
                        }`}
                        onDragOver={(event) => dragOver(event, `person:${person.name}`)}
                        onDragLeave={clearDrop}
                        onDrop={(event) => {
                            event.preventDefault();
                            const paths = pathsFromDrop(event);
                            clearDrop();
                            // Dropping onto an existing person extends their
                            // references rather than replacing them.
                            if (paths.length) onDropNewPerson(paths);
                        }}
                    >
                        <button
                            type="button"
                            className="tag-main"
                            onClick={() =>
                                onSelectPerson(selectedPerson === person.name ? null : person.name)
                            }
                            disabled={busy}
                        >
                            <span className="tag-name">{person.name}</span>
                            <span className="tag-count person">{person.files}</span>
                        </button>
                        <div className="tag-meta">
                            <span title="Reference faces">{person.references} ref</span>
                            <button
                                type="button"
                                className="tag-forget"
                                onClick={() => onForgetPerson(person.name)}
                                disabled={busy}
                                title={`Forget "${person.name}"`}
                            >
                                ×
                            </button>
                        </div>
                    </li>
                ))}
            </ul>

            <DropZone
                label="+ drop photos of someone to name them"
                active={dropTarget === 'new-person'}
                onDragOver={(event) => dragOver(event, 'new-person')}
                onDragLeave={clearDrop}
                onDrop={(event) => {
                    event.preventDefault();
                    const paths = pathsFromDrop(event);
                    clearDrop();
                    if (paths.length) onDropNewPerson(paths);
                }}
            />

            {clusters.length > 0 && (
                <>
                    <div className="rail-head">
                        <h2>Unnamed faces</h2>
                    </div>
                    <ul className="cluster-list">
                        {clusters.map((cluster) => (
                            <li key={cluster.face_ids[0]}>
                                <button
                                    type="button"
                                    className="cluster"
                                    onClick={() => onNameCluster(cluster)}
                                    disabled={busy}
                                    title="Give this person a name"
                                >
                                    {cluster.sample_path && (
                                        <img
                                            src={window.api.thumbUrl(cluster.sample_path, 96)}
                                            alt=""
                                        />
                                    )}
                                    <span>
                                        seen {cluster.size}× in {cluster.files} file
                                        {cluster.files === 1 ? '' : 's'}
                                    </span>
                                </button>
                            </li>
                        ))}
                    </ul>
                </>
            )}
        </aside>
    );
}

function DropZone({
    label,
    active,
    onDragOver,
    onDragLeave,
    onDrop,
}: {
    label: string;
    active: boolean;
    onDragOver: (event: React.DragEvent) => void;
    onDragLeave: () => void;
    onDrop: (event: React.DragEvent) => void;
}) {
    return (
        <div
            className={`new-tag ${active ? 'drop-active' : ''}`}
            onDragOver={onDragOver}
            onDragLeave={onDragLeave}
            onDrop={onDrop}
        >
            <span>{label}</span>
        </div>
    );
}
