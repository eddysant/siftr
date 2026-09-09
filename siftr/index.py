"""Building the library index: walk files, embed frames, detect faces."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .db import Database
from .embed import Embedder
from .faces import FaceAnalyzer, FaceRecognitionUnavailable, bbox_to_text, match
from .media import discover, frames

Progress = Callable[[str], None]


@dataclass
class IndexStats:
    scanned: int = 0
    indexed: int = 0
    skipped_unchanged: int = 0
    failed: int = 0
    frames_embedded: int = 0
    faces_found: int = 0
    faces_named: int = 0
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        parts = [
            f"{self.indexed} indexed",
            f"{self.skipped_unchanged} unchanged",
            f"{self.frames_embedded} frames embedded",
        ]
        if self.faces_found:
            parts.append(f"{self.faces_found} faces ({self.faces_named} named)")
        if self.failed:
            parts.append(f"{self.failed} failed")
        return ", ".join(parts)


def build_index(
    db: Database,
    root: Path,
    embedder: Embedder,
    *,
    video_samples: int = 8,
    detect_faces: bool = True,
    face_analyzer: FaceAnalyzer | None = None,
    force: bool = False,
    progress: Progress | None = None,
) -> IndexStats:
    """Index every media file under ``root``.

    Unchanged files are skipped by (path, size, mtime) so re-running on a large
    library is cheap. Any single file that fails to decode is recorded and
    skipped — one corrupt file must not abort a multi-hour scan.
    """
    stats = IndexStats()
    say = progress or (lambda _msg: None)

    if detect_faces and face_analyzer is None:
        face_analyzer = FaceAnalyzer()

    owners, reference_matrix = db.all_person_references()

    for media in discover(root):
        stats.scanned += 1

        if not force and db.is_unchanged(media.path, media.size, media.mtime_ns):
            stats.skipped_unchanged += 1
            continue

        say(f"[{stats.scanned}] {media.path.name}")

        try:
            extracted = frames(media.path, media.kind, samples=video_samples)
        except Exception as exc:
            stats.failed += 1
            stats.errors.append(f"{media.path}: {exc}")
            continue

        if not extracted:
            stats.failed += 1
            stats.errors.append(f"{media.path}: no decodable frames")
            continue

        try:
            vectors = embedder.embed_images([f.image for f in extracted])
        except Exception as exc:
            stats.failed += 1
            stats.errors.append(f"{media.path}: embedding failed: {exc}")
            continue

        file_id = db.upsert_file(media.path, media.kind, media.size, media.mtime_ns)
        db.add_embeddings(file_id, vectors, [f.time for f in extracted])
        stats.indexed += 1
        stats.frames_embedded += len(extracted)

        if detect_faces and face_analyzer is not None:
            try:
                found = _index_faces(
                    db, file_id, extracted, face_analyzer, owners, reference_matrix
                )
                stats.faces_found += found[0]
                stats.faces_named += found[1]
            except FaceRecognitionUnavailable:
                # Announce once, then continue indexing embeddings without faces.
                say("face recognition unavailable — install 'siftr[faces]'; continuing")
                detect_faces = False
            except Exception as exc:
                stats.errors.append(f"{media.path}: face detection failed: {exc}")

        # Commit per file so an interrupted scan keeps everything done so far.
        db.commit()

    db.commit()
    return stats


def _index_faces(
    db: Database,
    file_id: int,
    extracted,
    analyzer: FaceAnalyzer,
    owners: list[tuple[int, str]],
    reference_matrix: np.ndarray,
) -> tuple[int, int]:
    records = []
    named = 0
    for frame in extracted:
        for face in analyzer.detect(frame.image, frame.time):
            hit = match(face.vector, reference_matrix, owners)
            person_id, score = (None, None)
            if hit is not None:
                person_id, _name, score = hit
                named += 1
            records.append(
                {
                    "vector": face.vector,
                    "bbox": bbox_to_text(face.bbox),
                    "frame_time": face.frame_time,
                    "person_id": person_id,
                    "score": score,
                }
            )
    if records:
        db.add_file_faces(file_id, records)
    return len(records), named


def rematch_faces(db: Database, threshold: float | None = None) -> int:
    """Re-check every unidentified face against the current people list.

    Called after registering a new person so their photos are found without
    re-embedding the whole library — the expensive work is already stored.
    """
    from .faces import DEFAULT_MATCH_THRESHOLD

    cutoff = DEFAULT_MATCH_THRESHOLD if threshold is None else threshold
    owners, reference_matrix = db.all_person_references()
    if not owners:
        return 0

    assigned = 0
    for rows, matrix in db.iter_unassigned_faces():
        for row, vector in zip(rows, matrix, strict=True):
            hit = match(vector, reference_matrix, owners, threshold=cutoff)
            if hit is not None:
                person_id, _name, score = hit
                db.assign_face(int(row["id"]), person_id, score)
                assigned += 1
    db.commit()
    return assigned


def apply_concept(db: Database, name: str, threshold: float | None = None) -> int:
    """Score every indexed file against a concept and store the matches."""
    from .concepts import best_per_file
    from .vectors import cosine, from_blob

    row = db.get_concept(name)
    if row is None:
        raise KeyError(f"unknown concept: {name}")

    prototype = from_blob(row["prototype"])
    cutoff = float(row["threshold"]) if threshold is None else threshold

    scores_by_file: dict[int, float] = {}
    for rows, matrix in db.iter_embeddings():
        if matrix.shape[1] != prototype.shape[0]:
            raise ValueError(
                "Indexed embeddings have a different dimension than this concept "
                f"({matrix.shape[1]} vs {prototype.shape[0]}). The concept was taught "
                "with a different CLIP model — re-teach it, or re-index with the "
                "model the concept was built from."
            )
        batch = best_per_file([int(r["file_id"]) for r in rows], cosine(prototype, matrix))
        for file_id, score in batch.items():
            if score > scores_by_file.get(file_id, float("-inf")):
                scores_by_file[file_id] = score

    matches = [(fid, s) for fid, s in scores_by_file.items() if s >= cutoff]
    db.set_file_concepts(int(row["id"]), matches)
    return len(matches)
