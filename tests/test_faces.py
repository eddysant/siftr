"""Face matching logic and the face path through the indexer.

InsightFace itself is not exercised here — downloading a 280 MB model would make
CI slow and network-dependent. A stub analyzer supplies embeddings so the
matching, thresholding and storage logic is covered.
"""

from __future__ import annotations

import numpy as np
import pytest

from siftr.faces import DEFAULT_MATCH_THRESHOLD, DetectedFace, largest_face, match
from siftr.index import build_index
from siftr.vectors import normalize


def _face(seed, dim=16):
    rng = np.random.default_rng(seed)
    return normalize(rng.standard_normal(dim).astype(np.float32))


class StubAnalyzer:
    """Returns a fixed face for every frame, keyed by the caller's setup."""

    def __init__(self, vectors, bbox=(10, 10, 110, 110)):
        self.vectors = vectors
        self.bbox = bbox
        self.calls = 0

    def detect(self, image, frame_time=0.0):
        self.calls += 1
        return [
            DetectedFace(vector=v, bbox=self.bbox, det_score=0.99, frame_time=frame_time)
            for v in self.vectors
        ]


# ------------------------------------------------------------------- matching


def test_match_finds_the_right_person():
    alice, bob = _face(1), _face(2)
    owners = [(1, "Alice"), (2, "Bob")]
    result = match(bob, np.vstack([alice, bob]), owners)
    assert result is not None
    assert result[1] == "Bob"
    assert result[2] == pytest.approx(1.0)


def test_match_returns_none_below_threshold():
    reference = _face(1)
    stranger = normalize(-reference)
    assert match(stranger, reference, [(1, "Alice")]) is None


def test_match_with_no_references_is_none():
    assert match(_face(1), np.empty((0, 0), dtype=np.float32), []) is None


def test_match_takes_the_best_of_several_references_for_one_person():
    """Multiple reference photos per person is the point — the best one wins."""
    target = _face(3)
    poor = normalize(_face(4) * 0.1 + target * 0.2)
    owners = [(7, "Sam"), (7, "Sam")]
    result = match(target, np.vstack([poor, target]), owners)
    assert result is not None and result[0] == 7
    assert result[2] > 0.99


def test_match_honors_a_custom_threshold():
    a, b = _face(5), _face(6)
    similarity = float(normalize(a) @ normalize(b))
    assert match(b, a, [(1, "A")], threshold=similarity - 0.01) is not None
    assert match(b, a, [(1, "A")], threshold=similarity + 0.01) is None


def test_default_threshold_is_conservative():
    """Prefer a missed tag over a wrong one."""
    assert 0.2 < DEFAULT_MATCH_THRESHOLD < 0.6


# ---------------------------------------------------------------- face sizing


def test_largest_face_picks_the_biggest_box():
    small = DetectedFace(_face(1), (0, 0, 30, 30), 0.9)
    big = DetectedFace(_face(2), (0, 0, 200, 200), 0.9)
    assert largest_face([small, big]) is big


def test_largest_face_of_empty_is_none():
    assert largest_face([]) is None


def test_detected_face_size_uses_the_short_side():
    """A wide, shallow box is a bad detection, and its short side says so."""
    assert DetectedFace(_face(1), (0, 0, 300, 40), 0.9).size == 40


# ------------------------------------------------------- indexer's face path


def test_index_stores_detected_faces(db, tmp_path, make_images, embedder):
    make_images(tmp_path / "lib", (200, 150, 130), count=3)
    analyzer = StubAnalyzer([_face(1)])

    stats = build_index(db, tmp_path / "lib", embedder, detect_faces=True, face_analyzer=analyzer)
    assert stats.faces_found == 3
    assert stats.faces_named == 0  # nobody registered yet
    assert db.conn.execute("SELECT count(*) FROM file_faces").fetchone()[0] == 3


def test_index_names_faces_of_registered_people(db, tmp_path, make_images, embedder):
    sam = _face(21)
    person_id = db.upsert_person("Sam")
    db.replace_person_faces(person_id, sam, ["ref.jpg"])

    make_images(tmp_path / "lib", (200, 150, 130), count=2)
    stats = build_index(
        db, tmp_path / "lib", embedder, detect_faces=True, face_analyzer=StubAnalyzer([sam])
    )

    assert stats.faces_found == 2
    assert stats.faces_named == 2
    assert len(db.files_for_person("Sam")) == 2


def test_index_leaves_strangers_unnamed(db, tmp_path, make_images, embedder):
    person_id = db.upsert_person("Sam")
    reference = _face(21)
    db.replace_person_faces(person_id, reference, ["ref.jpg"])

    make_images(tmp_path / "lib", (200, 150, 130), count=2)
    stranger = normalize(-reference)
    stats = build_index(
        db,
        tmp_path / "lib",
        embedder,
        detect_faces=True,
        face_analyzer=StubAnalyzer([stranger]),
    )

    assert stats.faces_found == 2 and stats.faces_named == 0
    assert db.files_for_person("Sam") == []


def test_multiple_faces_per_image_are_all_stored(db, tmp_path, make_images, embedder):
    make_images(tmp_path / "lib", (200, 150, 130), count=1)
    analyzer = StubAnalyzer([_face(1), _face(2), _face(3)])
    stats = build_index(db, tmp_path / "lib", embedder, detect_faces=True, face_analyzer=analyzer)
    assert stats.faces_found == 3


def test_missing_face_extras_do_not_abort_indexing(db, tmp_path, make_images, embedder):
    """Embeddings must still be indexed when face deps are absent."""
    from siftr.faces import FaceRecognitionUnavailable

    class Unavailable:
        def detect(self, image, frame_time=0.0):
            raise FaceRecognitionUnavailable("not installed")

    make_images(tmp_path / "lib", (200, 150, 130), count=3)
    stats = build_index(
        db, tmp_path / "lib", embedder, detect_faces=True, face_analyzer=Unavailable()
    )
    assert stats.indexed == 3
    assert stats.frames_embedded == 3
    assert stats.faces_found == 0


def test_face_detection_error_is_recorded_not_fatal(db, tmp_path, make_images, embedder):
    class Broken:
        def detect(self, image, frame_time=0.0):
            raise RuntimeError("onnx exploded")

    make_images(tmp_path / "lib", (200, 150, 130), count=2)
    stats = build_index(db, tmp_path / "lib", embedder, detect_faces=True, face_analyzer=Broken())
    assert stats.indexed == 2
    assert len(stats.errors) == 2
    assert all("face detection failed" in e for e in stats.errors)
