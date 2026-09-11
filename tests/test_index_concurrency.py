"""The threaded indexing pipeline.

Decode, preprocess and face detection run on a pool; the model forward runs on
the calling thread in batches. These cover the properties that split has to
preserve — every file indexed exactly once, partial work kept, one bad file not
taking down the pool — and the ones it introduces.
"""

import threading

import pytest

from siftr.faces import FaceRecognitionUnavailable
from siftr.index import build_index


def test_every_file_is_indexed_exactly_once(db, tmp_path, make_images, embedder):
    make_images(tmp_path / "lib", (200, 40, 40), count=40)
    stats = build_index(db, tmp_path / "lib", embedder, detect_faces=False, workers=4)

    assert stats.indexed == 40
    assert db.count_files() == 40
    paths = [r["path"] for r in db.conn.execute("SELECT path FROM files")]
    assert len(set(paths)) == 40, "no file indexed twice"


def test_embeddings_stay_attached_to_the_right_file(db, tmp_path, make_images, embedder):
    """Out-of-order completion must not misalign vectors and paths."""
    from siftr.media import load_image
    from siftr.vectors import cosine, from_blob

    make_images(tmp_path / "lib", (200, 40, 40), count=12, jitter=20)
    build_index(db, tmp_path / "lib", embedder, detect_faces=False, workers=4)

    rows = db.conn.execute(
        "SELECT f.path, e.vector FROM files f JOIN embeddings e ON e.file_id = f.id"
    ).fetchall()
    assert len(rows) == 12
    for row in rows:
        expected = embedder.preprocess(load_image(row["path"]))
        stored = from_blob(row["vector"])
        assert cosine(expected, stored.reshape(1, -1))[0] > 0.999, row["path"]


def test_single_worker_matches_many_workers(db, tmp_path, make_images, embedder):
    from siftr.db import Database

    make_images(tmp_path / "lib", (30, 160, 90), count=15, jitter=10)
    build_index(db, tmp_path / "lib", embedder, detect_faces=False, workers=1)
    serial = {
        r["path"]: r["vector"]
        for r in db.conn.execute(
            "SELECT f.path, e.vector FROM files f JOIN embeddings e ON e.file_id = f.id"
        )
    }

    with Database(tmp_path / "parallel.db") as other:
        build_index(other, tmp_path / "lib", embedder, detect_faces=False, workers=8)
        parallel = {
            r["path"]: r["vector"]
            for r in other.conn.execute(
                "SELECT f.path, e.vector FROM files f JOIN embeddings e ON e.file_id = f.id"
            )
        }

    assert serial == parallel, "worker count must not change the result"


def test_one_unreadable_file_does_not_stop_the_pool(db, tmp_path, make_images, embedder):
    make_images(tmp_path / "lib", (200, 40, 40), count=20)
    (tmp_path / "lib" / "broken.png").write_bytes(b"not a png")

    stats = build_index(db, tmp_path / "lib", embedder, detect_faces=False, workers=4)

    assert stats.indexed == 20
    assert stats.failed == 1
    assert any("broken.png" in e for e in stats.errors)


def test_cancellation_stops_early_and_keeps_finished_work(db, tmp_path, make_images, embedder):
    make_images(tmp_path / "lib", (200, 40, 40), count=60)
    seen = threading.Event()

    def stop_after_a_few() -> bool:
        # Stop once a handful are done, without depending on exact timing.
        if db.count_files() >= 5:
            seen.set()
        return seen.is_set()

    stats = build_index(
        db,
        tmp_path / "lib",
        embedder,
        detect_faces=False,
        workers=4,
        should_stop=stop_after_a_few,
    )

    assert 0 < stats.indexed < 60, f"should have stopped early, indexed {stats.indexed}"
    assert db.count_files() == stats.indexed, "finished work must be committed"


def test_a_cancelled_scan_resumes(db, tmp_path, make_images, embedder):
    make_images(tmp_path / "lib", (200, 40, 40), count=30)
    stop = threading.Event()

    def stopper() -> bool:
        if db.count_files() >= 3:
            stop.set()
        return stop.is_set()

    first = build_index(
        db, tmp_path / "lib", embedder, detect_faces=False, workers=2, should_stop=stopper
    )
    second = build_index(db, tmp_path / "lib", embedder, detect_faces=False, workers=2)

    assert first.indexed + second.indexed == 30
    assert second.skipped_unchanged == first.indexed
    assert db.count_files() == 30


def test_missing_face_extras_keep_the_embeddings(db, tmp_path, make_images, embedder):
    """Losing a file's vectors because face support is absent would be worse
    than simply having no faces."""

    class Unavailable:
        def detect(self, image, frame_time=0.0):
            raise FaceRecognitionUnavailable("not installed")

    make_images(tmp_path / "lib", (200, 150, 130), count=12)
    stats = build_index(
        db, tmp_path / "lib", embedder, detect_faces=True, face_analyzer=Unavailable(), workers=4
    )

    assert stats.indexed == 12
    assert stats.frames_embedded == 12
    assert stats.faces_found == 0


def test_face_errors_are_recorded_but_keep_the_file(db, tmp_path, make_images, embedder):
    class Broken:
        def detect(self, image, frame_time=0.0):
            raise RuntimeError("onnx exploded")

    make_images(tmp_path / "lib", (200, 150, 130), count=6)
    stats = build_index(
        db, tmp_path / "lib", embedder, detect_faces=True, face_analyzer=Broken(), workers=4
    )

    assert stats.indexed == 6, "embeddings survive a face failure"
    assert len(stats.errors) == 6
    assert all("face detection failed" in e for e in stats.errors)


def test_faces_are_detected_across_threads(db, tmp_path, make_images, embedder):
    import numpy as np

    from siftr.faces import DetectedFace
    from siftr.vectors import normalize

    calls = []
    lock = threading.Lock()

    class Counting:
        def detect(self, image, frame_time=0.0):
            with lock:
                calls.append(threading.current_thread().name)
            rng = np.random.default_rng(len(calls))
            vector = normalize(rng.standard_normal(16).astype(np.float32))
            return [DetectedFace(vector, (0, 0, 100, 100), 0.9, frame_time)]

    make_images(tmp_path / "lib", (200, 150, 130), count=24)
    stats = build_index(
        db, tmp_path / "lib", embedder, detect_faces=True, face_analyzer=Counting(), workers=4
    )

    assert stats.faces_found == 24
    assert len(set(calls)) > 1, "detection should be spread across workers"


def test_workers_default_is_sane(db, tmp_path, make_images, embedder):
    make_images(tmp_path / "lib", (200, 40, 40), count=5)
    stats = build_index(db, tmp_path / "lib", embedder, detect_faces=False)
    assert stats.indexed == 5


def test_an_empty_folder_is_not_an_error(db, tmp_path):
    (tmp_path / "empty").mkdir()
    stats = build_index(db, tmp_path / "empty", None, detect_faces=False)
    assert stats.scanned == 0 and stats.indexed == 0


@pytest.mark.parametrize("workers", [1, 2, 8])
def test_progress_reports_each_file_once(db, tmp_path, make_images, embedder, workers):
    make_images(tmp_path / "lib", (200, 40, 40), count=16)
    messages = []
    lock = threading.Lock()

    def progress(message: str) -> None:
        with lock:
            messages.append(message)

    build_index(
        db, tmp_path / "lib", embedder, detect_faces=False, workers=workers, progress=progress
    )
    per_file = [m for m in messages if m.startswith("[")]
    assert len(per_file) == 16


def test_frame_vectors_keep_their_order_within_a_file(db, tmp_path, embedder, monkeypatch):
    """A video's frames must line up with their timestamps.

    Single-frame images cannot catch a reversal, so this uses a file that yields
    several visually distinct frames.
    """
    from PIL import Image

    import siftr.index as index_mod
    from siftr.media import Frame
    from siftr.vectors import cosine, from_blob

    colours = [(220, 20, 20), (20, 220, 20), (20, 20, 220), (220, 220, 20)]
    fake_frames = [Frame(Image.new("RGB", (32, 32), c), float(i)) for i, c in enumerate(colours)]

    clip = tmp_path / "lib" / "clip.mp4"
    clip.parent.mkdir(parents=True, exist_ok=True)
    clip.write_bytes(b"placeholder")

    monkeypatch.setattr(index_mod, "frames", lambda path, kind, samples=8: fake_frames)

    build_index(db, tmp_path / "lib", embedder, detect_faces=False, workers=2)

    rows = db.conn.execute(
        "SELECT frame_time, vector FROM embeddings ORDER BY frame_time"
    ).fetchall()
    assert len(rows) == 4
    for row, frame in zip(rows, fake_frames, strict=True):
        expected = embedder.preprocess(frame.image)
        stored = from_blob(row["vector"])
        assert cosine(expected, stored.reshape(1, -1))[0] > 0.999, (
            f"frame at t={row['frame_time']} carries the wrong vector"
        )


def test_vectors_are_never_written_to_another_files_row(db, tmp_path, make_images, embedder):
    """Cross-file misalignment: the failure that concurrency makes possible."""
    from siftr.media import load_image
    from siftr.vectors import cosine, from_blob

    # Strongly distinct colours so a swap between any two files is unmissable.
    for i, colour in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]):
        make_images(tmp_path / "lib" / f"g{i}", colour, count=4, jitter=0)

    build_index(db, tmp_path / "lib", embedder, detect_faces=False, workers=8)

    rows = db.conn.execute(
        "SELECT f.path, e.vector FROM files f JOIN embeddings e ON e.file_id = f.id"
    ).fetchall()
    assert len(rows) == 16
    for row in rows:
        expected = embedder.preprocess(load_image(row["path"]))
        stored = from_blob(row["vector"])
        assert cosine(expected, stored.reshape(1, -1))[0] > 0.999, row["path"]
