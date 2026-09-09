"""Searching the index: by taught concept, by ad-hoc example folder, or by text."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .concepts import best_per_file, example_images
from .db import Database
from .embed import Embedder
from .vectors import centroid, cosine, from_blob


@dataclass
class Hit:
    path: Path
    kind: str
    score: float


def _search_vector(
    db: Database, query: np.ndarray, limit: int, threshold: float | None
) -> list[Hit]:
    """Rank every indexed file against one query vector."""
    best: dict[int, float] = {}
    meta: dict[int, tuple[str, str]] = {}

    for rows, matrix in db.iter_embeddings():
        if matrix.shape[1] != query.shape[0]:
            raise ValueError(
                f"Query dimension {query.shape[0]} does not match the index "
                f"({matrix.shape[1]}). Both must come from the same CLIP model."
            )
        scores = cosine(query, matrix)
        file_ids = [int(r["file_id"]) for r in rows]
        for row, file_id in zip(rows, file_ids, strict=True):
            meta[file_id] = (row["path"], row["kind"])
        for file_id, score in best_per_file(file_ids, scores).items():
            if score > best.get(file_id, float("-inf")):
                best[file_id] = score

    hits = [
        Hit(Path(meta[fid][0]), meta[fid][1], score)
        for fid, score in best.items()
        if threshold is None or score >= threshold
    ]
    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[:limit]


def by_concept(
    db: Database, name: str, limit: int = 50, threshold: float | None = None
) -> list[Hit]:
    """Search using a previously taught concept.

    Reads the stored prototype directly, so this works without re-embedding and
    without an ``apply`` pass having been run.
    """
    row = db.get_concept(name)
    if row is None:
        raise KeyError(f"unknown concept: {name}")
    cutoff = float(row["threshold"]) if threshold is None else threshold
    return _search_vector(db, from_blob(row["prototype"]), limit, cutoff)


def by_examples(
    db: Database,
    folder: Path,
    embedder: Embedder,
    limit: int = 50,
    threshold: float | None = None,
) -> list[Hit]:
    """One-shot search from a folder of examples, without saving a concept.

    This is the "drop files in a folder and find more like them" path: no naming,
    no threshold tuning, just ranked results.
    """
    paths = example_images(folder)
    kept, vectors = embedder.embed_paths(paths)
    if len(kept) == 0:
        raise ValueError(f"none of the images in {folder} could be read")
    return _search_vector(db, centroid(vectors), limit, threshold)


def by_text(
    db: Database,
    prompt: str,
    embedder: Embedder,
    limit: int = 50,
    threshold: float | None = None,
) -> list[Hit]:
    """Search with a text prompt via CLIP's shared image/text space.

    Note that image-image and text-image similarities are *not* on the same
    scale — CLIP text scores run much lower — so a threshold tuned for example
    search will reject everything here. Left as ``None`` by default for that
    reason: rank, don't threshold.
    """
    query = embedder.embed_text([prompt])[0]
    return _search_vector(db, query, limit, threshold)


def by_person(db: Database, name: str, limit: int = 50) -> list[Hit]:
    """Every file containing a registered person, best match first."""
    rows = db.files_for_person(name, limit=limit)
    return [Hit(Path(r["path"]), r["kind"], float(r["score"] or 0.0)) for r in rows]
