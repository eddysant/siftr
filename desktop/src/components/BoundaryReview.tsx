import { useEffect, useState } from 'react';
import * as api from '../api';
import type { BoundaryFile } from '../types';

interface Props {
    tag: string;
    onClose: () => void;
    onChanged: () => void;
}

/**
 * Confirm or reject the files a tag is least sure about.
 *
 * Counter-examples are the highest-leverage input a tag can take — measured on a
 * real attribute set, five took recall from 5/8 to 7/8 where nearly doubling the
 * positives did nothing. They were also the most awkward thing to provide: you
 * had to go and find them. This puts the useful ones in front of you.
 *
 * Only near-misses are shown. A photo scoring nowhere near the threshold teaches
 * the tag nothing it does not already know.
 */
export function BoundaryReview({ tag, onClose, onChanged }: Props) {
    const [files, setFiles] = useState<BoundaryFile[]>([]);
    const [threshold, setThreshold] = useState(0);
    const [busy, setBusy] = useState(false);
    const [note, setNote] = useState('');
    const [error, setError] = useState<string | null>(null);

    const load = async () => {
        try {
            const result = await api.getBoundary(tag);
            setFiles(result.files);
            setThreshold(result.threshold);
        } catch (err) {
            setError(String(err));
        }
    };

    useEffect(() => {
        void load();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [tag]);

    const answer = async (file: BoundaryFile, isMatch: boolean) => {
        setBusy(true);
        setError(null);
        try {
            const result = await api.reviewBoundary(tag, file.path, isMatch);
            setNote(
                `${isMatch ? 'kept' : 'rejected'} ${file.name} — threshold now ` +
                    `${result.threshold.toFixed(3)}, ${result.rejections} counter-example` +
                    `${result.rejections === 1 ? '' : 's'}`,
            );
            // The line moved, so which files sit near it has changed too.
            await load();
            onChanged();
        } catch (err) {
            setError(String(err));
        } finally {
            setBusy(false);
        }
    };

    return (
        <div className="modal-backdrop" onMouseDown={onClose}>
            <div className="review" onMouseDown={(event) => event.stopPropagation()}>
                <header>
                    <h3>Is this “{tag}”?</h3>
                    <span className="muted">
                        the {files.length} photos closest to the line (threshold{' '}
                        {threshold.toFixed(3)})
                    </span>
                    <button type="button" onClick={onClose}>
                        Done
                    </button>
                </header>

                {error && <p className="banner error">{error}</p>}
                {note && !error && <p className="review-note">{note}</p>}

                <ul className="review-grid">
                    {files.map((file) => (
                        <li key={file.path} className={file.matching ? 'is-match' : ''}>
                            <img src={window.api.thumbUrl(file.path, 240)} alt={file.name} />
                            <span className="review-score">
                                {file.score.toFixed(3)} {file.matching ? '· tagged' : '· not tagged'}
                            </span>
                            <div className="review-actions">
                                <button
                                    type="button"
                                    className="yes"
                                    disabled={busy}
                                    onClick={() => answer(file, true)}
                                    title="Yes — this is the tag"
                                >
                                    ✓ yes
                                </button>
                                <button
                                    type="button"
                                    className="no"
                                    disabled={busy}
                                    onClick={() => answer(file, false)}
                                    title="No — teach the tag to exclude this"
                                >
                                    ✗ no
                                </button>
                            </div>
                        </li>
                    ))}
                </ul>

                {files.length === 0 && !error && (
                    <p className="muted">Nothing to review — index a library and teach a tag first.</p>
                )}
            </div>
        </div>
    );
}
