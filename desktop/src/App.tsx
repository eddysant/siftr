import { useCallback, useEffect, useMemo, useState } from 'react';
import * as api from './api';
import { Grid } from './components/Grid';
import { TagRail } from './components/TagRail';
import { filterFiles, tagCounts, toggleTag } from './filter';
import type { Job, MatchMode, MediaFile, Tag } from './types';

export default function App() {
    const [files, setFiles] = useState<MediaFile[]>([]);
    const [tags, setTags] = useState<Tag[]>([]);
    const [roots, setRoots] = useState<string[]>([]);
    const [selectedTags, setSelectedTags] = useState<string[]>([]);
    const [mode, setMode] = useState<MatchMode>('any');
    const [query, setQuery] = useState('');
    const [untaggedOnly, setUntaggedOnly] = useState(false);
    const [chosen, setChosen] = useState<Set<string>>(new Set());
    const [job, setJob] = useState<Job | null>(null);
    const [status, setStatus] = useState('');
    const [error, setError] = useState<string | null>(null);

    const busy = job?.state === 'running';

    const refresh = useCallback(async () => {
        try {
            const [library, tagList] = await Promise.all([api.getLibrary(), api.getTags()]);
            setFiles(library.files);
            setRoots(library.roots);
            setTags(tagList.tags);
        } catch (err) {
            setError(String(err));
        }
    }, []);

    useEffect(() => {
        window.api.onServiceError(setError);
        void refresh();
    }, [refresh]);

    const run = useCallback(
        async (start: () => Promise<Job>, label: string) => {
            setError(null);
            try {
                const started = await start();
                const finished = await api.followJob(started.id, setJob);
                setJob(finished);
                if (finished.state === 'failed') setError(finished.error ?? `${label} failed`);
                else setStatus(`${label} finished`);
                await refresh();
            } catch (err) {
                setError(String(err));
            }
        },
        [refresh],
    );

    const openFolder = async () => {
        const folder = await window.api.chooseFolder();
        if (folder) await run(() => api.startIndex(folder), 'Indexing');
    };

    const teach = async (name: string, paths: string[]) => {
        setError(null);
        try {
            const result = await api.teachTag(name, paths);
            setStatus(
                `“${result.name}” learned from ${result.examples} example${
                    result.examples === 1 ? '' : 's'
                } (threshold ${result.threshold.toFixed(2)})`,
            );
            await refresh();
            // Teaching a tag is only useful once the library has been scored
            // against it, so that follows immediately rather than waiting for
            // the user to find a separate button.
            await run(() => api.startScore(true), 'Scoring');
        } catch (err) {
            setError(String(err));
        }
    };

    const toggleFileTag = async (file: MediaFile, tag: string) => {
        // A chip on a photo is the model's decision; clicking it is a correction.
        try {
            await api.setOverride(file.path, tag, 'off');
            setStatus(`removed “${tag}” from ${file.name}`);
            await run(() => api.startScore(true), 'Scoring');
        } catch (err) {
            setError(String(err));
        }
    };

    const undo = async () => {
        try {
            const result = await api.undoRenames();
            setStatus(`restored ${result.restored} filename${result.restored === 1 ? '' : 's'}`);
            await refresh();
        } catch (err) {
            setError(String(err));
        }
    };

    const counts = useMemo(() => tagCounts(files), [files]);
    const visible = useMemo(
        () => filterFiles(files, { query, selectedTags, mode, untaggedOnly }),
        [files, query, selectedTags, mode, untaggedOnly],
    );

    return (
        <div className="app">
            <header className="bar">
                <span className="brand">siftr</span>
                <button type="button" onClick={openFolder} disabled={busy}>
                    Open folder…
                </button>
                <button
                    type="button"
                    onClick={() => run(() => api.startScore(true), 'Scoring')}
                    disabled={busy || !tags.length}
                    title="Re-score the library and write tags into filenames"
                >
                    Re-score
                </button>
                <button type="button" onClick={undo} disabled={busy || !roots.length}>
                    Undo renames
                </button>

                <input
                    className="search"
                    value={query}
                    placeholder="filter by filename"
                    onChange={(event) => setQuery(event.target.value)}
                />
                <label className="check">
                    <input
                        type="checkbox"
                        checked={untaggedOnly}
                        onChange={(event) => setUntaggedOnly(event.target.checked)}
                    />
                    untagged only
                </label>

                <span className="spacer" />
                <span className="count">
                    {visible.length} / {files.length}
                </span>
            </header>

            {busy && (
                <div className="progress">
                    {job?.message || job?.kind}
                    {job?.total ? ` — ${job.current}/${job.total}` : ` — ${job?.current ?? 0}`}
                </div>
            )}
            {error && (
                <div className="banner error" onClick={() => setError(null)}>
                    {error}
                </div>
            )}
            {!error && status && !busy && (
                <div className="banner" onClick={() => setStatus('')}>
                    {status}
                </div>
            )}

            <div className="body">
                <TagRail
                    tags={tags}
                    counts={counts}
                    selected={selectedTags}
                    mode={mode}
                    busy={busy}
                    onToggle={(name) => setSelectedTags((s) => toggleTag(s, name))}
                    onModeChange={setMode}
                    onDropFiles={teach}
                    onCreateTag={teach}
                    onForget={async (name) => {
                        await api.forgetTag(name);
                        setSelectedTags((s) => s.filter((t) => t !== name));
                        await refresh();
                    }}
                />
                <Grid
                    files={visible}
                    selected={chosen}
                    onSelect={(path, additive) =>
                        setChosen((current) => {
                            const next = additive ? new Set(current) : new Set<string>();
                            if (additive && current.has(path)) next.delete(path);
                            else next.add(path);
                            return next;
                        })
                    }
                    onToggleTag={toggleFileTag}
                    onReveal={(path) => window.api.revealInFinder(path)}
                />
            </div>
        </div>
    );
}
