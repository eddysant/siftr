import { useCallback, useEffect, useMemo, useState } from 'react';
import * as api from './api';
import { Grid } from './components/Grid';
import { NamePrompt } from './components/NamePrompt';
import { BoundaryReview } from './components/BoundaryReview';
import { Duplicates } from './components/Duplicates';
import { TagRail } from './components/TagRail';
import { filterFiles, tagCounts, toggleTag } from './filter';
import type {
    FaceCluster,
    Job,
    MatchMode,
    MediaFile,
    OrganizeMode,
    Person,
    Tag,
} from './types';

/** What the name prompt is currently collecting a name for. */
type Pending =
    | { kind: 'tag'; paths: string[] }
    | { kind: 'person'; paths: string[] }
    | { kind: 'cluster'; cluster: FaceCluster }
    | null;

export default function App() {
    const [files, setFiles] = useState<MediaFile[]>([]);
    const [tags, setTags] = useState<Tag[]>([]);
    const [people, setPeople] = useState<Person[]>([]);
    const [clusters, setClusters] = useState<FaceCluster[]>([]);
    const [roots, setRoots] = useState<string[]>([]);

    const [selectedTags, setSelectedTags] = useState<string[]>([]);
    const [selectedPerson, setSelectedPerson] = useState<string | null>(null);
    const [personFiles, setPersonFiles] = useState<Set<string> | null>(null);
    const [mode, setMode] = useState<MatchMode>('any');
    const [query, setQuery] = useState('');
    const [untaggedOnly, setUntaggedOnly] = useState(false);

    const [chosen, setChosen] = useState<Set<string>>(new Set());
    const [job, setJob] = useState<Job | null>(null);
    const [pendingNames, setPendingNames] = useState(0);
    const [organizeMode, setOrganizeMode] = useState<OrganizeMode>('rename');
    const [prompt, setPrompt] = useState<Pending>(null);
    const [reviewing, setReviewing] = useState<string | null>(null);
    const [showDuplicates, setShowDuplicates] = useState(false);
    const [status, setStatus] = useState('');
    const [error, setError] = useState<string | null>(null);
    // Kept separate from `error`: when the Python service never started, every
    // request fails too, and those generic failures would otherwise bury the one
    // message that actually tells the user what to do.
    const [fatal, setFatal] = useState<string | null>(null);

    const busy = job?.state === 'running';

    const refresh = useCallback(async () => {
        try {
            const [library, tagList, peopleList, pending, settings] = await Promise.all([
                api.getLibrary(),
                api.getTags(),
                api.getPeople(),
                api.pendingRenames(),
                api.getSettings(),
            ]);
            setFiles(library.files);
            setRoots(library.roots);
            setTags(tagList.tags);
            setPeople(peopleList.people);
            setPendingNames(pending.pending);
            setOrganizeMode(settings.organize_mode);
            // Clusters are only meaningful once faces exist, and the call is
            // cheap enough to fold into the same refresh.
            try {
                setClusters((await api.getClusters()).clusters);
            } catch {
                setClusters([]);
            }
        } catch (err) {
            setError(String(err));
        }
    }, []);

    useEffect(() => {
        window.api.onServiceError(setFatal);
        void refresh();
    }, [refresh]);

    useEffect(() => {
        // The menu drives the same handlers as the toolbar rather than
        // duplicating their logic in the main process.
        window.api.onMenuAction((action) => {
            switch (action) {
                case 'open-folder':
                    void openFolder();
                    break;
                case 'rescore':
                    void score();
                    break;
                case 'organize':
                    void applyOrganize();
                    break;
                case 'undo-changes':
                    void undo();
                    break;
                case 'reveal':
                    for (const path of chosen) void window.api.revealInFinder(path);
                    break;
                case 'clear-selection':
                    setChosen(new Set());
                    break;
                case 'mode-rename':
                    void changeMode('rename');
                    break;
                case 'mode-move':
                    void changeMode('move');
                    break;
                case 'mode-off':
                    void changeMode('off');
                    break;
            }
        });
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [chosen, organizeMode, tags.length, roots.length]);

    const run = useCallback(
        async (start: () => Promise<Job>, label: string) => {
            setError(null);
            try {
                const started = await start();
                const finished = await api.followJob(started.id, setJob);
                setJob(finished);
                if (finished.state === 'failed') setError(finished.error ?? `${label} failed`);
                else if (finished.state === 'cancelled') setStatus(`${label} cancelled`);
                else setStatus(`${label} finished`);
                await refresh();
            } catch (err) {
                setError(String(err));
            }
        },
        [refresh],
    );

    const score = useCallback(() => run(() => api.startScore(true), 'Scoring'), [run]);

    const openFolder = async () => {
        const folder = await window.api.chooseFolder();
        if (folder) await run(() => api.startIndex(folder), 'Indexing');
    };

    const teach = async (name: string, paths: string[], negatives: string[] = []) => {
        setError(null);
        try {
            const result = await api.teachTag(name, paths, negatives);
            setStatus(
                `“${result.name}” learned from ${result.examples} example${
                    result.examples === 1 ? '' : 's'
                }${negatives.length ? ` and ${negatives.length} counter-example(s)` : ''} ` +
                    `(threshold ${result.threshold.toFixed(2)})`,
            );
            await score();
        } catch (err) {
            setError(String(err));
        }
    };

    const dropOnTag = (name: string, paths: string[], negative: boolean) =>
        negative ? teach(name, [], paths) : teach(name, paths);

    const registerPerson = async (name: string, paths: string[]) => {
        setError(null);
        try {
            const result = await api.addPerson(name, paths);
            setStatus(`“${result.name}” registered with ${result.references} reference face(s)`);
            const matched = await window.api.request<{ identified: number }>(
                'POST',
                '/api/people/rematch',
            );
            if (matched.identified) setStatus(`found ${matched.identified} more face(s)`);
            await refresh();
        } catch (err) {
            setError(String(err));
        }
    };

    const confirmPrompt = async (name: string) => {
        const current = prompt;
        setPrompt(null);
        if (!current) return;
        if (current.kind === 'tag') await teach(name, current.paths);
        else if (current.kind === 'person') await registerPerson(name, current.paths);
        else {
            try {
                await api.nameCluster(name, current.cluster.face_ids);
                setStatus(`named ${current.cluster.size} face(s) “${name}”`);
                await refresh();
            } catch (err) {
                setError(String(err));
            }
        }
    };

    const removeTagFromFile = async (file: MediaFile, tag: string) => {
        try {
            await api.setOverride(file.path, tag, 'off');
            setStatus(`removed “${tag}” from ${file.name}`);
            await score();
        } catch (err) {
            setError(String(err));
        }
    };

    /** Apply a tag to every selected photo at once. */
    const batchTag = async (tag: string) => {
        const paths = Array.from(chosen);
        if (!paths.length) return;
        try {
            await Promise.all(paths.map((path) => api.setOverride(path, tag, 'on')));
            setStatus(`pinned “${tag}” on ${paths.length} photo(s)`);
            await score();
        } catch (err) {
            setError(String(err));
        }
    };

    const changeMode = async (mode: OrganizeMode) => {
        try {
            await api.setOrganizeMode(mode);
            setOrganizeMode(mode);
            setStatus(
                mode === 'rename'
                    ? 'matches will be renamed with [tag] in place'
                    : mode === 'move'
                      ? 'matches will be filed into each tag’s folder'
                      : 'files will be left alone',
            );
            await refresh();
        } catch (err) {
            setError(String(err));
        }
    };

    const chooseDestination = async (tag: string, inverse: boolean) => {
        const folder = await window.api.chooseFolder();
        if (!folder) return;
        try {
            await api.setDestination(tag, folder, inverse);
            setStatus(
                inverse
                    ? `anything that is not “${tag}” files into ${folder}`
                    : `“${tag}” files into ${folder}`,
            );
            await refresh();
        } catch (err) {
            setError(String(err));
        }
    };

    const applyOrganize = async () => {
        setError(null);
        try {
            const result = await api.organizeNow();
            const verb = result.mode === 'move' ? 'filed' : 'renamed';
            setStatus(
                result.changed
                    ? `${verb} ${result.changed} file${result.changed === 1 ? '' : 's'}` +
                          (result.paired ? ` (${result.paired} paired group(s) kept together)` : '')
                    : 'nothing to change',
            );
            if (result.errors.length) setError(result.errors.join('\n'));
            await refresh();
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

    const selectPerson = async (name: string | null) => {
        setSelectedPerson(name);
        if (!name) {
            setPersonFiles(null);
            return;
        }
        try {
            const result = await api.personFiles(name);
            setPersonFiles(new Set(result.files.map((f) => f.path)));
        } catch (err) {
            setError(String(err));
        }
    };

    const counts = useMemo(() => tagCounts(files), [files]);
    const visible = useMemo(() => {
        const byTag = filterFiles(files, { query, selectedTags, mode, untaggedOnly });
        return personFiles ? byTag.filter((file) => personFiles.has(file.path)) : byTag;
    }, [files, query, selectedTags, mode, untaggedOnly, personFiles]);

    return (
        <div className="app">
            <header className="bar">
                <span className="brand">siftr</span>
                <button type="button" onClick={openFolder} disabled={busy}>
                    Open folder…
                </button>
                <button type="button" onClick={score} disabled={busy || !tags.length}>
                    Re-score
                </button>
                <label className="mode-select" title="What happens to files that match a tag">
                    <select
                        value={organizeMode}
                        disabled={busy}
                        onChange={(event) => changeMode(event.target.value as OrganizeMode)}
                    >
                        <option value="rename">rename with [tag]</option>
                        <option value="move">move to tag folder</option>
                        <option value="off">leave files alone</option>
                    </select>
                </label>
                <button
                    type="button"
                    onClick={() => setShowDuplicates(true)}
                    disabled={busy || !files.length}
                    title="Find duplicate and near-duplicate files"
                >
                    Duplicates
                </button>
                <button type="button" onClick={undo} disabled={busy || !roots.length}>
                    Undo
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
                {chosen.size > 0 && (
                    <span className="selection">
                        {chosen.size} selected
                        <select
                            value=""
                            disabled={busy || !tags.length}
                            onChange={(event) => {
                                if (event.target.value) void batchTag(event.target.value);
                            }}
                        >
                            <option value="">add to tag…</option>
                            {tags.map((tag) => (
                                <option key={tag.name} value={tag.name}>
                                    {tag.name}
                                </option>
                            ))}
                        </select>
                        <button type="button" onClick={() => setChosen(new Set())}>
                            clear
                        </button>
                    </span>
                )}
                <span className="count">
                    {visible.length} / {files.length}
                </span>
            </header>

            {busy && (
                <div className="progress">
                    <span>
                        {job?.message || job?.kind}
                        {job?.total ? ` — ${job.current}/${job.total}` : ` — ${job?.current ?? 0}`}
                    </span>
                    <button
                        type="button"
                        onClick={() => job && api.cancelJob(job.id).catch(() => {})}
                        title="Stop after the current file"
                    >
                        Cancel
                    </button>
                </div>
            )}

            {!busy && organizeMode === 'rename' && pendingNames > 0 && (
                <div className="banner warn">
                    {pendingNames} filename{pendingNames === 1 ? '' : 's'} no longer match their
                    tags.
                    <button type="button" onClick={score}>
                        Apply
                    </button>
                </div>
            )}

            {fatal && <div className="banner error fatal">{fatal}</div>}
            {!fatal && error && (
                <div className="banner error" onClick={() => setError(null)}>
                    {error}
                </div>
            )}
            {!fatal && !error && status && !busy && (
                <div className="banner" onClick={() => setStatus('')}>
                    {status}
                </div>
            )}

            <div className="body">
                <TagRail
                    tags={tags}
                    people={people}
                    clusters={clusters}
                    counts={counts}
                    selectedTags={selectedTags}
                    selectedPerson={selectedPerson}
                    mode={mode}
                    busy={busy}
                    onToggleTag={(name) => setSelectedTags((s) => toggleTag(s, name))}
                    onSelectPerson={selectPerson}
                    onModeChange={setMode}
                    onDropOnTag={dropOnTag}
                    onDropNewTag={(paths) => setPrompt({ kind: 'tag', paths })}
                    onDropNewPerson={(paths) => setPrompt({ kind: 'person', paths })}
                    onForgetTag={async (name) => {
                        await api.forgetTag(name);
                        setSelectedTags((s) => s.filter((t) => t !== name));
                        await refresh();
                    }}
                    onForgetPerson={async (name) => {
                        await api.forgetPerson(name);
                        if (selectedPerson === name) await selectPerson(null);
                        await refresh();
                    }}
                    organizeMode={organizeMode}
                    onSetDestination={chooseDestination}
                    onReviewTag={setReviewing}
                    onNameCluster={(cluster) => setPrompt({ kind: 'cluster', cluster })}
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
                    onToggleTag={removeTagFromFile}
                    onReveal={(path) => window.api.revealInFinder(path)}
                />
            </div>

            {showDuplicates && <Duplicates onClose={() => setShowDuplicates(false)} />}

            {reviewing && (
                <BoundaryReview
                    tag={reviewing}
                    onClose={() => setReviewing(null)}
                    onChanged={() => void score()}
                />
            )}

            {prompt && (
                <NamePrompt
                    title={
                        prompt.kind === 'tag'
                            ? 'Name this tag'
                            : prompt.kind === 'person'
                              ? 'Who is this?'
                              : 'Name this person'
                    }
                    detail={
                        prompt.kind === 'cluster'
                            ? `${prompt.cluster.size} faces across ${prompt.cluster.files} file(s)`
                            : `${prompt.paths.length} photo(s)`
                    }
                    confirmLabel={prompt.kind === 'tag' ? 'Teach' : 'Name'}
                    onConfirm={confirmPrompt}
                    onCancel={() => setPrompt(null)}
                />
            )}
        </div>
    );
}
