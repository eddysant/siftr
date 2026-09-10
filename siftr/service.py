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

from .concepts import calibrate_threshold
from .db import Database
from .embed import Embedder
from .jobs import Job
from .naming import validate_tag
from .rename import apply_renames, plan_renames, undo_last
from .vectors import centroid, cosine, from_blob


def _library_scores(db: Database, prototype: np.ndarray) -> np.ndarray:
    """Every indexed embedding's similarity to a prototype.

    The negative pool for threshold calibration. Returns an empty array when
    nothing is indexed yet, or when the index was built with a different model,
    in which case the caller falls back to a positives-only rule.
    """
    chunks: list[np.ndarray] = []
    for _rows, matrix in db.iter_embeddings():
        if matrix.shape[1] != prototype.shape[0]:
            return np.empty(0, dtype=np.float32)
        chunks.append(cosine(prototype, matrix))
    return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32)


@dataclass
class TeachResult:
    name: str
    n_examples: int
    threshold: float
    cohesion: float


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
    kept, vectors = embedder.embed_paths([Path(p) for p in paths])
    if len(kept) == 0:
        raise ValueError("none of the dropped files could be read as images")

    prototype = centroid(vectors)
    scores = cosine(prototype, vectors)
    cohesion = float(scores.mean())
    # No explicit negatives is the normal case when files are dropped onto a
    # tag, so the indexed library stands in as the negative pool.
    threshold = calibrate_threshold(scores, _library_scores(db, prototype))

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


def apply_tags_to_filenames(db: Database, root: Path | None = None, dry_run: bool = False) -> dict:
    """Write each file's effective tags into its name on disk.

    Runs after scoring. Every batch is journaled so it can be undone, and the
    index is updated in step so no embedding is orphaned.
    """
    known = [row["name"] for row in db.list_concepts()]
    effective = db.effective_tags()

    tagged: list[tuple[Path, list[str]]] = []
    for row in db.library_view():
        path = Path(row["path"])
        tagged.append((path, effective.get(int(row["id"]), [])))

    plans = plan_renames(tagged, known)
    result = apply_renames(plans, db=db, root=root, dry_run=dry_run)
    return {
        "renamed": result.count,
        "skipped": result.skipped,
        "errors": result.errors,
        "changes": [[str(s), str(t)] for s, t in result.applied[:200]],
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
