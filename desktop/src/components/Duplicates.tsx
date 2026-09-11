import { useEffect, useState } from 'react';
import * as api from '../api';
import type { DuplicateGroup } from '../types';

interface Props {
    onClose: () => void;
}

/**
 * Duplicate and near-duplicate groups.
 *
 * Nothing here deletes anything. Each group names the copy worth keeping and
 * lists the rest as redundant; acting on that is a separate, deliberate step,
 * because a duplicate finder that deletes is a duplicate finder you have to be
 * certain about, and near-duplicate detection is a heuristic.
 */
export function Duplicates({ onClose }: Props) {
    const [groups, setGroups] = useState<DuplicateGroup[]>([]);
    const [total, setTotal] = useState(0);
    const [distance, setDistance] = useState(0.12);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    const load = async (value: number) => {
        setLoading(true);
        setError(null);
        try {
            const result = await api.getDuplicates(value);
            setGroups(result.groups);
            setTotal(result.redundant_total);
        } catch (err) {
            setError(String(err));
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        void load(distance);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [distance]);

    return (
        <div className="modal-backdrop" onMouseDown={onClose}>
            <div className="dupes" onMouseDown={(event) => event.stopPropagation()}>
                <header>
                    <h3>Duplicates</h3>
                    <span className="muted">
                        {loading
                            ? 'scanning…'
                            : `${groups.length} group${groups.length === 1 ? '' : 's'}, ` +
                              `${total} redundant file${total === 1 ? '' : 's'}`}
                    </span>
                    {/* Range goes well past the default on purpose. How far apart
                        duplicates land depends on the library — measured across two
                        fixtures, resize-and-recompress variants sat at 0.8% of bits
                        in one and 13% in another, because per-pixel noise survives
                        resizing differently. Unrelated photos stayed above 45% in
                        both, so there is always a gap; the slider is how you find
                        where yours is. */}
                    <label className="dupes-slider" title="How different two photos may be">
                        strictness
                        <input
                            type="range"
                            min={2}
                            max={35}
                            value={Math.round(distance * 100)}
                            onChange={(event) => setDistance(Number(event.target.value) / 100)}
                        />
                        <span className="muted">{(distance * 100).toFixed(0)}%</span>
                    </label>
                    <button type="button" onClick={onClose}>
                        Done
                    </button>
                </header>

                {error && <p className="banner error">{error}</p>}

                {!loading && groups.length === 0 && !error && (
                    <p className="muted">No duplicates found at this strictness.</p>
                )}

                {groups.map((group) => (
                    <section key={group.keeper + group.paths.length} className="dupe-group">
                        <h4>
                            {group.paths.length} files ·{' '}
                            {group.kind === 'exact'
                                ? 'identical'
                                : `${(group.distance * 100).toFixed(1)}% differ`}
                        </h4>
                        <ul>
                            {group.paths.map((path) => {
                                const keep = path === group.keeper;
                                return (
                                    <li key={path} className={keep ? 'keeper' : ''}>
                                        <img src={window.api.thumbUrl(path, 200)} alt="" />
                                        <span className="dupe-label">
                                            {keep ? 'keep' : 'redundant'}
                                        </span>
                                        <span
                                            className="dupe-path"
                                            title={path}
                                            onClick={() => window.api.revealInFinder(path)}
                                        >
                                            {path.split('/').pop()}
                                        </span>
                                    </li>
                                );
                            })}
                        </ul>
                    </section>
                ))}

                <p className="muted dupes-foot">
                    Nothing is deleted. Double-click a filename to reveal it in Finder, or use{' '}
                    <code>siftr duplicates -o ~/review</code> to collect the redundant copies.
                </p>
            </div>
        </div>
    );
}
