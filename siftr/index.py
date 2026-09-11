"""Building the library index: walk files, embed frames, detect faces."""

from __future__ import annotations

import os
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .db import Database
from .duplicates import content_hash, perceptual_hash
from .embed import Embedder
from .faces import FaceAnalyzer, FaceRecognitionUnavailable, bbox_to_text, match
from .media import MediaFile, discover, frames

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


@dataclass(kw_only=True)
class _Prepared:
    """One file, decoded and made ready for the model.

    Carries tensors rather than images on purpose: a preprocessed 224x224 tensor
    is ~600 KB where the 12 MP photo it came from is ~36 MB, which is what makes
    a pipeline deep enough to keep the GPU fed affordable in memory.
    """

    media: MediaFile
    tensors: list
    frame_times: list[float]
    faces: list[dict]
    content_hash: bytes | None = None
    phash: bytes | None = None
    pixels: int | None = None
    error: str | None = None
    #: Set when the face extras are not installed. The file's embeddings are
    #: still valid and must still be stored — losing them because face support
    #: is absent would be a far worse failure than simply having no faces.
    faces_unavailable: bool = False


def _prepare(
    media: MediaFile,
    embedder: Embedder,
    video_samples: int,
    analyzer: FaceAnalyzer | None,
) -> _Prepared:
    """Decode, preprocess and face-detect one file.

    Runs on a worker thread. Every step here releases the GIL — image codecs,
    torchvision's resize, and ONNX Runtime all do — which is what makes
    threading worth anything in Python. Measured on 12 MP files: decode 2.5x on
    four threads, preprocessing 4.2x, face detection 3.6x.

    Errors are returned rather than raised so one unreadable file cannot take
    down the pool.
    """
    try:
        extracted = frames(media.path, media.kind, samples=video_samples)
    except Exception as exc:
        return _Prepared(
            media=media, tensors=[], frame_times=[], faces=[], error=f"{media.path}: {exc}"
        )

    if not extracted:
        return _Prepared(
            media=media,
            tensors=[],
            frame_times=[],
            faces=[],
            error=f"{media.path}: no decodable frames",
        )

    try:
        tensors = [embedder.preprocess(f.image) for f in extracted]
    except Exception as exc:
        return _Prepared(
            media=media,
            tensors=[],
            frame_times=[],
            faces=[],
            error=f"{media.path}: preprocessing failed: {exc}",
        )

    # Duplicate fingerprints, taken from the first frame — for a video that is a
    # poster frame, which is what makes two encodes of the same clip comparable.
    # Both are cheap next to the model work and ride along on the worker thread.
    first = extracted[0].image
    try:
        digest = content_hash(media.path)
        fingerprint = perceptual_hash(first)
        pixels = first.size[0] * first.size[1]
    except Exception:
        # Losing a fingerprint costs duplicate detection for this file, nothing
        # else; the embeddings are still perfectly good.
        digest, fingerprint, pixels = None, None, None

    faces: list[dict] = []
    if analyzer is not None:
        try:
            for frame in extracted:
                for face in analyzer.detect(frame.image, frame.time):
                    faces.append(
                        {
                            "vector": face.vector,
                            "bbox": bbox_to_text(face.bbox),
                            "frame_time": face.frame_time,
                        }
                    )
        except FaceRecognitionUnavailable:
            return _Prepared(
                media=media,
                tensors=tensors,
                frame_times=[f.time for f in extracted],
                faces=[],
                content_hash=digest,
                phash=fingerprint,
                pixels=pixels,
                faces_unavailable=True,
            )
        except Exception as exc:
            # A face failure must not cost the file its embeddings.
            return _Prepared(
                media=media,
                tensors=tensors,
                frame_times=[f.time for f in extracted],
                faces=[],
                content_hash=digest,
                phash=fingerprint,
                pixels=pixels,
                error=f"{media.path}: face detection failed: {exc}",
            )

    return _Prepared(
        media=media,
        tensors=tensors,
        frame_times=[f.time for f in extracted],
        faces=faces,
        content_hash=digest,
        phash=fingerprint,
        pixels=pixels,
    )


def build_index(
    db: Database,
    root: Path,
    embedder: Embedder,
    *,
    video_samples: int = 8,
    detect_faces: bool = True,
    face_analyzer: FaceAnalyzer | None = None,
    force: bool = False,
    workers: int | None = None,
    progress: Progress | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> IndexStats:
    """Index every media file under ``root``.

    Decoding, preprocessing and face detection run on a thread pool; the model
    forward pass runs on the calling thread in batches. That split is deliberate:
    the GPU work is a small fraction of the total (~8% of the embedding stage)
    and does not parallelise usefully, while everything feeding it is CPU work
    that does.

    Unchanged files are skipped by (path, size, mtime, samples) so re-running on
    a large library is cheap. Any single file that fails to decode is recorded
    and skipped — one corrupt file must not abort a multi-hour scan.

    ``should_stop`` is polled between files; when it returns True the scan stops
    and everything already written is kept.
    """
    stats = IndexStats()
    say = progress or (lambda _msg: None)
    pool_size = workers if workers is not None else min(8, (os.cpu_count() or 4))

    if detect_faces and face_analyzer is None:
        face_analyzer = FaceAnalyzer()
    analyzer = face_analyzer if detect_faces else None

    owners, reference_matrix = db.all_person_references()

    pending: list[MediaFile] = []
    for media in discover(root):
        stats.scanned += 1
        wanted = video_samples if media.kind == "video" else 1
        if not force and db.is_unchanged(media.path, media.size, media.mtime_ns, wanted):
            stats.skipped_unchanged += 1
            continue
        pending.append(media)

    if not pending:
        db.commit()
        return stats

    say(f"{len(pending)} file(s) to index on {pool_size} worker(s)")

    def write(prepared: _Prepared) -> None:
        """Store one prepared file. Called only from this thread — SQLite
        connections are not safe to share across threads."""
        if prepared.error:
            stats.errors.append(prepared.error)
            if not prepared.tensors:
                stats.failed += 1
                return

        vectors = embedder.embed_tensors(prepared.tensors)
        media = prepared.media
        file_id = db.upsert_file(
            media.path,
            media.kind,
            media.size,
            media.mtime_ns,
            len(prepared.tensors),
            content_hash=prepared.content_hash,
            phash=prepared.phash,
            pixels=prepared.pixels,
        )
        db.add_embeddings(file_id, vectors, prepared.frame_times)
        stats.indexed += 1
        stats.frames_embedded += len(prepared.tensors)

        if prepared.faces:
            named = 0
            for face in prepared.faces:
                hit = match(face["vector"], reference_matrix, owners)
                if hit is not None:
                    face["person_id"], _name, face["score"] = hit
                    named += 1
                else:
                    face["person_id"], face["score"] = None, None
            db.add_file_faces(file_id, prepared.faces)
            stats.faces_found += len(prepared.faces)
            stats.faces_named += named

    # A bounded window of in-flight work: enough to keep every worker busy
    # without decoding the whole library into memory at once.
    window = pool_size * 4
    cancelled = False

    with ThreadPoolExecutor(max_workers=pool_size, thread_name_prefix="siftr-prep") as pool:
        queued = deque()
        upcoming = iter(pending)
        done = 0

        def submit_more() -> None:
            while len(queued) < window:
                try:
                    media = next(upcoming)
                except StopIteration:
                    return
                queued.append(pool.submit(_prepare, media, embedder, video_samples, analyzer))

        submit_more()
        while queued:
            future = queued.popleft()
            try:
                prepared = future.result()
            except Exception as exc:
                stats.failed += 1
                stats.errors.append(str(exc))
                continue

            if prepared.faces_unavailable and analyzer is not None:
                # Announce once, stop asking for faces on work not yet
                # submitted, and carry on indexing embeddings. Files already
                # in flight still return their tensors, so nothing is lost.
                say("face recognition unavailable — install 'siftr[faces]'; continuing")
                analyzer = None

            write(prepared)
            done += 1
            say(f"[{done}/{len(pending)}] {prepared.media.path.name}")

            # Commit periodically rather than per file: an fsync per photo
            # dominates once decoding is no longer the bottleneck, and a batch
            # still bounds how much an interrupted scan loses.
            if done % 50 == 0:
                db.commit()

            if should_stop is not None and should_stop():
                cancelled = True
                break
            submit_more()

        if cancelled:
            for future in queued:
                future.cancel()

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
