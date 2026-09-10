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

    Returns the effective tag set per file (model output with user overrides
    applied), which is what gets written into filenames.
    """
    concepts = db.list_concepts()
    if job:
        job.total = len(concepts)

    for i, row in enumerate(concepts):
        if job and job.cancelled:
            break
        if job:
            job.current = i
            job.message = f"scoring {row['name']}"

        prototype = from_blob(row["prototype"])
        cutoff = float(row["threshold"])
        best: dict[int, float] = {}

        for rows, matrix in db.iter_embeddings():
            if matrix.shape[1] != prototype.shape[0]:
                raise ValueError(
                    f"tag '{row['name']}' was taught with a different CLIP model "
                    f"({prototype.shape[0]}-d) than the index holds "
                    f"({matrix.shape[1]}-d); re-teach it or re-index"
                )
            scores = cosine(prototype, matrix)
            for r, score in zip(rows, scores, strict=True):
                file_id = int(r["file_id"])
                value = float(score)
                if value > best.get(file_id, float("-inf")):
                    best[file_id] = value

        db.set_file_concepts(int(row["id"]), [(fid, s) for fid, s in best.items() if s >= cutoff])

    if job:
        job.current = len(concepts)
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
