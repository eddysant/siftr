"""The operations the UI performs, independent of HTTP.

Keeping these here rather than in the route handlers means the whole app is
testable without spinning up a server, and the CLI could grow the same features
without duplicating logic.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .companions import find_companions, group_paths, tags_for_groups
from .concepts import MIN_THRESHOLD, calibrate_threshold
from .db import Database
from .embed import Embedder
from .jobs import Job
from .naming import validate_tag
from .rename import apply_renames, plan_moves, plan_renames, undo_last
from .vectors import centroid, cosine, from_blob


def _library_scores(
    db: Database, prototype: np.ndarray, exclude: Sequence[Path] = ()
) -> np.ndarray:
    """Every indexed embedding's similarity to a prototype.

    The negative pool for threshold calibration. ``exclude`` must carry the files
    the prototype was built from, and leaving them in is not a rounding error —
    it breaks calibration outright. Examples are usually *in* the library, where
    they necessarily score highest, so the widest gap in the distribution becomes
    "my examples versus everything else". The resulting threshold then matches
    the examples and nothing more. Measured on a 27-photo library: leaving them
    in gave a threshold of 0.786 and 0/8 recall on held-out matches.

    Returns an empty array when nothing else is indexed, or when the index was
    built with a different model, in which case the caller falls back to a
    positives-only rule.
    """
    skip = {str(Path(p).resolve()) for p in exclude}
    chunks: list[np.ndarray] = []
    for rows, matrix in db.iter_embeddings():
        if matrix.shape[1] != prototype.shape[0]:
            return np.empty(0, dtype=np.float32)
        scores = cosine(prototype, matrix)
        if skip:
            keep = [i for i, r in enumerate(rows) if str(Path(r["path"]).resolve()) not in skip]
            scores = scores[keep]
        if len(scores):
            chunks.append(scores)
    return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32)


@dataclass
class TeachResult:
    name: str
    n_examples: int
    threshold: float
    cohesion: float


def _retighten_with_negatives(
    db: Database, name: str, negatives: Sequence[Path], embedder: Embedder
) -> TeachResult:
    """Move an existing tag's threshold above a set of counter-examples.

    The prototype is left untouched: the examples still define what the tag looks
    like. Only the cutoff moves, to just above the most tag-like of the things
    the user has said are not it.
    """
    row = db.get_concept(name)
    if row is None:
        raise ValueError(
            f"“{name}” does not exist yet — drop examples of it before counter-examples"
        )

    prototype = from_blob(row["prototype"])
    _kept, negative_vectors = embedder.embed_paths([Path(p) for p in negatives])
    if not len(negative_vectors):
        raise ValueError("none of the counter-examples could be read as images")

    negative_scores = cosine(prototype, negative_vectors)
    # Counter-examples are a lower bound on the cutoff, never an upper one:
    # saying "not this" can only tighten a tag. Taking the max with the existing
    # threshold stops a set of very dissimilar negatives — whose strongest score
    # is low — from computing a *looser* cutoff and widening the tag.
    threshold = max(float(row["threshold"]), float(negative_scores.max()) + 0.01, MIN_THRESHOLD)

    db.save_concept(name, prototype, threshold, int(row["n_examples"]))
    return TeachResult(name, int(row["n_examples"]), threshold, 0.0)


def teach_from_paths(
    db: Database,
    name: str,
    paths: Sequence[Path],
    embedder: Embedder,
    negatives: Sequence[Path] = (),
) -> TeachResult:
    """Learn (or re-learn) a tag from an explicit list of example files.

    This is what dropping photos onto a tag calls. Unlike the CLI's folder-based
    path, examples arrive as a list, so the prototype is built here rather than
    by walking a directory.
    """
    name = validate_tag(name)

    # Anything the user has explicitly rejected for this tag is a counter-example
    # for good, not just for the teach that recorded it. Folding them in here
    # means a re-teach from fresh examples cannot quietly undo a correction.
    existing = db.get_concept(name)
    if existing is not None:
        stored = [Path(p) for p in db.rejections(int(existing["id"]))]
        negatives = list(dict.fromkeys([*negatives, *stored]))

    if not paths and negatives:
        # Counter-examples only: the user dropped "this is NOT that" onto an
        # existing tag. Saying what a tag excludes should not require restating
        # what it includes, so the prototype is kept and only the cutoff moves.
        return _retighten_with_negatives(db, name, negatives, embedder)

    kept, vectors = embedder.embed_paths([Path(p) for p in paths])
    if len(kept) == 0:
        raise ValueError("none of the dropped files could be read as images")

    prototype = centroid(vectors)
    scores = cosine(prototype, vectors)
    cohesion = float(scores.mean())
    # No explicit negatives is the normal case when files are dropped onto a
    # tag, so the indexed library stands in as the negative pool.
    threshold = calibrate_threshold(scores, _library_scores(db, prototype, kept))

    if negatives:
        _, negative_vectors = embedder.embed_paths([Path(p) for p in negatives])
        if len(negative_vectors):
            negative_scores = cosine(prototype, negative_vectors)
            # An explicit negative that genuinely resembles the tag must not
            # push the threshold past every positive.
            threshold = min(float(negative_scores.max()) + 0.01, float(np.percentile(scores, 10)))

    db.save_concept(name, prototype, threshold, len(kept))
    return TeachResult(name, len(kept), threshold, cohesion)


def example_paths_for(db: Database, name: str) -> list[str]:
    """Files a user has explicitly pinned to a tag, usable as extra examples."""
    row = db.get_concept(name)
    if row is None:
        return []
    return db.overrides_for_concept(int(row["id"]), "on")


def review_boundary_file(
    db: Database,
    name: str,
    path: Path,
    is_match: bool,
    embedder: Embedder,
) -> TeachResult:
    """Record a yes/no answer about one boundary file and re-learn the tag.

    A "no" is stored as a rejection and folded into the negatives on every future
    teach; a "yes" pins the tag on that file and makes it an example. Both survive
    re-teaching, which is the point — the whole value of an answer is that it
    keeps applying.
    """
    row = db.get_concept(name)
    if row is None:
        raise KeyError(f"unknown concept: {name}")
    concept_id = int(row["id"])
    path = Path(path)

    if is_match:
        db.clear_rejection(concept_id, path)
        file_id = db.file_id_for_path(path)
        if file_id is not None:
            db.set_override(file_id, concept_id, "on")
    else:
        db.add_rejection(concept_id, path)
        file_id = db.file_id_for_path(path)
        if file_id is not None:
            db.set_override(file_id, concept_id, "off")

    return relearn(db, name, embedder)


def relearn(db: Database, name: str, embedder: Embedder) -> TeachResult:
    """Re-teach a tag from everything currently known about it.

    Positives are the files pinned on; negatives are the rejections. Called after
    a boundary review so one answer immediately moves the threshold.
    """
    row = db.get_concept(name)
    if row is None:
        raise KeyError(f"unknown concept: {name}")
    concept_id = int(row["id"])

    positives = [Path(p) for p in db.overrides_for_concept(concept_id, "on")]
    negatives = [Path(p) for p in db.rejections(concept_id)]

    if not positives:
        # Nothing pinned: keep the prototype and move only the cutoff, the same
        # path an Alt-drop of counter-examples takes.
        if negatives:
            return _retighten_with_negatives(db, name, negatives, embedder)
        return TeachResult(name, int(row["n_examples"]), float(row["threshold"]), 0.0)

    return teach_from_paths(db, name, positives, embedder, negatives=negatives)


def verify_tag(
    db: Database,
    name: str,
    phrase: str | None = None,
    candidates: int = 200,
    progress=None,
) -> list[dict]:
    """Re-rank a tag's best candidates with open-vocabulary detection.

    Two stages by necessity. Detection is ~99x slower than embedding, so it
    cannot run over a library; CLIP picks the plausible few hundred out of the
    index that already exists and this checks only those. Bounded work for a
    much sharper answer.

    Returns every candidate with both scores, best-verified first — the CLIP
    score is kept so a disagreement between the two is visible rather than
    silently resolved.
    """
    from .grounding import verify_all
    from .search import by_concept

    row = db.get_concept(name)
    if row is None:
        raise KeyError(f"unknown concept: {name}")

    query = phrase or row["verify_phrase"] or name.replace("-", " ")
    # A generous candidate set, not just what already clears the threshold.
    # Verification exists to settle the uncertain cases, and the files the tag
    # already accepts are the least uncertain ones there are — passing the
    # stored cutoff here would verify only what needed it least.
    hits = by_concept(db, name, limit=candidates, threshold=0.0)
    if not hits:
        return []

    clip_scores = {hit.path: hit.score for hit in hits}
    verdicts = verify_all([h.path for h in hits], [query], progress=progress)
    return [
        {
            "path": str(v.path),
            "name": v.path.name,
            "verified": v.score,
            "clip": clip_scores.get(v.path, 0.0),
            "box": list(v.box) if v.box else None,
        }
        for v in verdicts
    ]


def boundary_files(db: Database, name: str, limit: int = 12) -> list[dict]:
    """The files nearest a tag's decision boundary, closest first.

    Counter-examples are the highest-leverage input a tag can get — measured on a
    real attribute set, five of them took recall from 5/8 to 7/8 where nearly
    doubling the positives did nothing. But they are only useful if they are
    *near-misses*: a photo that scores nowhere near the threshold teaches it
    nothing it did not already know.

    So this returns the files the tag is least certain about, on both sides of
    the line. Confirming or rejecting those is what moves a threshold.
    """
    row = db.get_concept(name)
    if row is None:
        raise KeyError(f"unknown concept: {name}")

    prototype = from_blob(row["prototype"])
    threshold = float(row["threshold"])

    best: dict[int, float] = {}
    meta: dict[int, tuple[str, str]] = {}
    for rows, matrix in db.iter_embeddings():
        if matrix.shape[1] != prototype.shape[0]:
            return []
        scores = cosine(prototype, matrix)
        for r, score in zip(rows, scores, strict=True):
            file_id = int(r["file_id"])
            meta[file_id] = (r["path"], r["kind"])
            value = float(score)
            if value > best.get(file_id, float("-inf")):
                best[file_id] = value

    ranked = sorted(best.items(), key=lambda kv: abs(kv[1] - threshold))
    out = []
    for file_id, score in ranked[:limit]:
        path, kind = meta[file_id]
        out.append(
            {
                "path": path,
                "name": Path(path).name,
                "kind": kind,
                "score": score,
                "matching": score >= threshold,
                "distance": abs(score - threshold),
            }
        )
    return out


def score_library(db: Database, job: Job | None = None) -> dict[int, list[str]]:
    """Score every indexed file against every tag and store the matches.

    One pass over the embeddings scoring *all* tags at once, rather than a pass
    per tag. The dot product is the cheap part; streaming several hundred
    thousand vectors out of SQLite once per tag was the expensive part, and with
    T tags that was T times more I/O and deserialization than necessary. Stacking
    the prototypes into a matrix turns the whole thing into one matmul per batch.

    Returns the effective tag set per file (model output with user overrides
    applied), which is what gets written into filenames.
    """
    concepts = db.list_concepts()
    if not concepts:
        return db.effective_tags()

    prototypes = np.vstack([from_blob(row["prototype"]) for row in concepts])
    thresholds = np.array([float(row["threshold"]) for row in concepts])

    if job:
        job.total = 0  # counted in embedding batches, which we do not know upfront
        job.message = f"scoring {len(concepts)} tag(s)"

    # best[concept_index] -> {file_id: best score}
    best: list[dict[int, float]] = [{} for _ in concepts]

    for rows, matrix in db.iter_embeddings():
        if job and job.cancelled:
            break
        if matrix.shape[1] != prototypes.shape[1]:
            raise ValueError(
                f"the index holds {matrix.shape[1]}-d embeddings but a tag was "
                f"taught at {prototypes.shape[1]}-d; re-teach the tags or re-index"
            )

        # (n_embeddings x dim) @ (dim x n_concepts) -> (n_embeddings x n_concepts)
        scores = matrix @ prototypes.T
        file_ids = [int(r["file_id"]) for r in rows]

        for concept_index in range(len(concepts)):
            column = scores[:, concept_index]
            target = best[concept_index]
            for file_id, score in zip(file_ids, column, strict=True):
                value = float(score)
                if value > target.get(file_id, float("-inf")):
                    target[file_id] = value

        if job:
            job.current += len(rows)

    if not (job and job.cancelled):
        for concept_index, row in enumerate(concepts):
            cutoff = thresholds[concept_index]
            db.set_file_concepts(
                int(row["id"]),
                [(fid, s) for fid, s in best[concept_index].items() if s >= cutoff],
            )

    return db.effective_tags()


ORGANIZE_MODES = ("rename", "move", "off")


def _tagged_groups(
    db: Database,
) -> tuple[list, dict[Path, list[str]], dict[Path, dict[str, float]]]:
    """Every indexed file grouped with its companions, with the group's tags.

    Companion grouping is what stops a Live Photo's still and motion clip being
    renamed or filed apart: the two halves match different tags on their own, so
    every member takes the primary's tags.
    """
    effective = db.effective_tags()
    rows = db.library_view()

    indexed = [Path(row["path"]) for row in rows]
    tags_by_path = {Path(row["path"]): effective.get(int(row["id"]), []) for row in rows}

    # Include companions that were never indexed — a .AAE sidecar or a Live
    # Photo's .MOV may not be a file siftr embeds, but it must still travel.
    everything: set[Path] = set(indexed)
    for path in indexed:
        everything.update(find_companions(path))

    groups = group_paths(everything)
    resolved = tags_for_groups(groups, tags_by_path)

    scores: dict[Path, dict[str, float]] = {}
    for name in (row["name"] for row in db.list_concepts()):
        for hit in db.files_for_concept(name):
            scores.setdefault(Path(hit["path"]), {})[name] = float(hit["score"])

    return groups, resolved, scores


def organize(
    db: Database,
    mode: str | None = None,
    root: Path | None = None,
    dry_run: bool = False,
) -> dict:
    """Apply the library's organize policy: rename in place, or file into folders.

    ``rename`` writes each file's tags into its name. ``move`` files it into the
    destination folder of its highest-scoring tag that has one. ``off`` leaves
    the filesystem alone entirely.

    Both modes act on companion groups rather than individual files, and both
    journal to the same manifest, so one undo covers either.
    """
    mode = mode or db.get_setting("organize_mode", "rename")
    if mode not in ORGANIZE_MODES:
        raise ValueError(f"mode must be one of {ORGANIZE_MODES}, got {mode!r}")
    if mode == "off":
        return {"mode": mode, "changed": 0, "skipped": 0, "errors": [], "changes": []}

    groups, resolved, scores = _tagged_groups(db)
    tagged = [(path, tags) for path, tags in resolved.items()]
    notes: list[str] = []

    if mode == "rename":
        known = [row["name"] for row in db.list_concepts()]
        plans = plan_renames(tagged, known)
    else:
        destinations = db.destinations()
        inverse = db.destinations(inverse=True)
        if not destinations and not inverse:
            return {
                "mode": mode,
                "changed": 0,
                "skipped": 0,
                "errors": ["no tag has a destination folder set"],
                "changes": [],
            }
        plans, notes = plan_moves(tagged, destinations, scores, inverse)

    result = apply_renames(plans, db=db, root=root, dry_run=dry_run)
    return {
        "mode": mode,
        "changed": result.count,
        "skipped": result.skipped,
        "already_filed": len(result.already_filed),
        "errors": [*result.errors, *notes],
        "paired": sum(1 for g in groups if g.is_paired),
        "changes": [[str(a), str(b)] for a, b in result.applied[:200]],
    }


def apply_tags_to_filenames(db: Database, root: Path | None = None, dry_run: bool = False) -> dict:
    """Rename-mode organize. Kept as a named entry point for the pending check."""
    out = organize(db, mode="rename", root=root, dry_run=dry_run)
    return {
        "renamed": out["changed"],
        "skipped": out["skipped"],
        "errors": out["errors"],
        "changes": out["changes"],
    }


def undo_renames(db: Database, root: Path) -> dict:
    result = undo_last(root, db=db)
    return {"restored": result.count, "errors": result.errors}


@dataclass
class PersonResult:
    name: str
    n_references: int
    skipped: list[str]


def register_person_from_paths(
    db: Database, name: str, paths: Sequence[Path], all_faces: bool = False
) -> PersonResult:
    """Register (or extend) a person from dropped photos of them.

    Takes the largest face in each photo by default: reference folders routinely
    contain group shots where the person is the one in front, and taking every
    face would pollute the reference set with whoever else was there.
    """
    from .faces import FaceAnalyzer, largest_face
    from .media import load_image

    name = validate_tag(name)
    analyzer = FaceAnalyzer()
    vectors: list[np.ndarray] = []
    sources: list[str] = []
    skipped: list[str] = []

    for path in (Path(p) for p in paths):
        try:
            found = analyzer.detect(load_image(path))
        except Exception as exc:
            skipped.append(f"{path.name}: {exc}")
            continue
        if not found:
            skipped.append(f"{path.name}: no face detected")
            continue
        for face in found if all_faces else [largest_face(found)]:
            vectors.append(face.vector)
            sources.append(str(path))

    if not vectors:
        raise ValueError(f"no faces found in the {len(list(paths))} dropped file(s)")

    person_id = db.upsert_person(name)
    # Keep any references already registered: dropping more photos onto a person
    # should extend their coverage across ages and angles, not replace it.
    existing = db.person_reference_vectors(person_id)
    existing_sources = db.person_reference_sources(person_id)
    if len(existing):
        vectors = [*existing, *vectors]
        sources = [*existing_sources, *sources]

    db.replace_person_faces(person_id, np.vstack(vectors), sources)
    return PersonResult(name, len(vectors), skipped)


def cluster_unnamed_faces(db: Database, threshold: float = 0.5, min_size: int = 3) -> list[dict]:
    """Group unidentified faces into probable people.

    Answers "who is this person appearing 40 times?" without asking for a name
    first. Uses single-link agglomeration over cosine similarity: a face joins a
    cluster when it is close to *any* member, which is what tolerates a person
    drifting across pose and lighting within one identity.

    The threshold is deliberately looser than the naming threshold — this only
    proposes groups for a human to name, so over-merging is cheaper than
    fragmenting one person into six clusters.
    """
    from .vectors import normalize

    rows_and_matrix = list(db.iter_unassigned_faces())
    if not rows_and_matrix:
        return []
    rows, matrix = rows_and_matrix[0]
    if len(rows) < min_size:
        return []

    matrix = normalize(matrix)
    similarity = matrix @ matrix.T

    # Union-find over pairs above the threshold.
    parent = list(range(len(rows)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    close = np.argwhere(similarity >= threshold)
    for i, j in close:
        if i < j:
            union(int(i), int(j))

    groups: dict[int, list[int]] = {}
    for index in range(len(rows)):
        groups.setdefault(find(index), []).append(index)

    clusters = []
    for members in groups.values():
        if len(members) < min_size:
            continue
        file_ids = {int(rows[m]["file_id"]) for m in members}
        clusters.append(
            {
                "size": len(members),
                "files": len(file_ids),
                "face_ids": [int(rows[m]["id"]) for m in members],
                # A representative face, for showing the user who this is.
                "sample_file_id": int(rows[members[0]]["file_id"]),
            }
        )
    clusters.sort(key=lambda c: c["size"], reverse=True)
    return clusters


def name_cluster(db: Database, face_ids: Sequence[int], name: str) -> PersonResult:
    """Turn a proposed cluster into a named person.

    The clustered faces themselves become that person's references, so naming a
    cluster immediately teaches the matcher without asking for reference photos.
    """
    name = validate_tag(name)
    vectors = db.face_vectors(face_ids)
    if not len(vectors):
        raise ValueError("no such faces")

    person_id = db.upsert_person(name)
    existing = db.person_reference_vectors(person_id)
    sources = db.person_reference_sources(person_id)
    combined = np.vstack([*existing, *vectors]) if len(existing) else vectors
    db.replace_person_faces(person_id, combined, [*sources, *["cluster"] * len(vectors)])

    for face_id in face_ids:
        db.assign_face(int(face_id), person_id, 1.0)
    db.commit()
    return PersonResult(name, len(combined), [])
