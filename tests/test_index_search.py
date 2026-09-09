"""End-to-end coverage of index → teach → apply → search, with a fake embedder."""

import numpy as np
import pytest

from siftr import search as search_mod
from siftr.concepts import learn
from siftr.index import apply_concept, build_index, rematch_faces
from siftr.vectors import normalize


def _index(db, folder, embedder, **kwargs):
    kwargs.setdefault("detect_faces", False)
    return build_index(db, folder, embedder, **kwargs)


def test_index_embeds_every_image(db, tmp_path, make_images, embedder):
    make_images(tmp_path / "lib", (200, 40, 40), count=5)
    stats = _index(db, tmp_path / "lib", embedder)
    assert stats.indexed == 5
    assert stats.frames_embedded == 5
    assert db.count_files() == 5


def test_reindex_skips_unchanged_files(db, tmp_path, make_images, embedder):
    make_images(tmp_path / "lib", (200, 40, 40), count=4)
    _index(db, tmp_path / "lib", embedder)
    again = _index(db, tmp_path / "lib", embedder)
    assert again.indexed == 0
    assert again.skipped_unchanged == 4


def test_force_reindexes_everything(db, tmp_path, make_images, embedder):
    make_images(tmp_path / "lib", (200, 40, 40), count=3)
    _index(db, tmp_path / "lib", embedder)
    forced = _index(db, tmp_path / "lib", embedder, force=True)
    assert forced.indexed == 3
    assert db.conn.execute("SELECT count(*) FROM embeddings").fetchone()[0] == 3


def test_unreadable_file_is_recorded_not_fatal(db, tmp_path, make_images, embedder):
    make_images(tmp_path / "lib", (200, 40, 40), count=2)
    (tmp_path / "lib" / "broken.png").write_bytes(b"nope")

    stats = _index(db, tmp_path / "lib", embedder)
    assert stats.indexed == 2
    assert stats.failed == 1
    assert any("broken.png" in e for e in stats.errors)


def test_search_by_examples_ranks_similar_first(db, tmp_path, make_images, embedder):
    make_images(tmp_path / "lib" / "red", (230, 20, 20), count=4, jitter=3)
    make_images(tmp_path / "lib" / "blue", (20, 20, 230), count=4, jitter=3)
    _index(db, tmp_path / "lib", embedder)

    make_images(tmp_path / "query", (230, 20, 20), count=2, jitter=2)
    hits = search_mod.by_examples(db, tmp_path / "query", embedder, limit=4)

    assert len(hits) == 4
    assert all("red" in str(h.path) for h in hits), [str(h.path) for h in hits]


def test_search_hits_are_sorted_descending(db, tmp_path, make_images, embedder):
    make_images(tmp_path / "lib" / "red", (230, 20, 20), count=3, jitter=3)
    make_images(tmp_path / "lib" / "blue", (20, 20, 230), count=3, jitter=3)
    _index(db, tmp_path / "lib", embedder)
    make_images(tmp_path / "q", (230, 20, 20), count=1)

    scores = [h.score for h in search_mod.by_examples(db, tmp_path / "q", embedder, limit=6)]
    assert scores == sorted(scores, reverse=True)


def test_search_respects_the_limit(db, tmp_path, make_images, embedder):
    make_images(tmp_path / "lib", (200, 40, 40), count=10)
    _index(db, tmp_path / "lib", embedder)
    make_images(tmp_path / "q", (200, 40, 40), count=1)
    assert len(search_mod.by_examples(db, tmp_path / "q", embedder, limit=3)) == 3


def test_teach_then_search_by_concept(db, tmp_path, make_images, embedder):
    make_images(tmp_path / "lib" / "red", (230, 20, 20), count=4, jitter=3)
    make_images(tmp_path / "lib" / "blue", (20, 20, 230), count=4, jitter=3)
    _index(db, tmp_path / "lib", embedder)

    make_images(tmp_path / "examples", (230, 20, 20), count=5, jitter=3)
    concept = learn("reds", tmp_path / "examples", embedder, negatives=tmp_path / "lib" / "blue")
    db.save_concept(concept.name, concept.prototype, concept.threshold, concept.n_examples)

    hits = search_mod.by_concept(db, "reds", limit=20)
    assert hits, "taught concept should match its own kind"
    assert all("red" in str(h.path) for h in hits)


def test_apply_concept_persists_tags(db, tmp_path, make_images, embedder):
    make_images(tmp_path / "lib" / "red", (230, 20, 20), count=3, jitter=3)
    make_images(tmp_path / "lib" / "blue", (20, 20, 230), count=3, jitter=3)
    _index(db, tmp_path / "lib", embedder)

    make_images(tmp_path / "examples", (230, 20, 20), count=4, jitter=3)
    concept = learn("reds", tmp_path / "examples", embedder, negatives=tmp_path / "lib" / "blue")
    db.save_concept(concept.name, concept.prototype, concept.threshold, concept.n_examples)

    count = apply_concept(db, "reds")
    assert count == len(db.files_for_concept("reds")) == count
    assert all("red" in r["path"] for r in db.files_for_concept("reds"))


def test_apply_unknown_concept_raises(db):
    with pytest.raises(KeyError):
        apply_concept(db, "nope")


def test_search_unknown_concept_raises(db):
    with pytest.raises(KeyError):
        search_mod.by_concept(db, "nope")


def test_dimension_mismatch_gives_an_actionable_error(db, tmp_path, make_images, embedder):
    """Switching CLIP models invalidates stored vectors; say so instead of crashing."""
    make_images(tmp_path / "lib", (200, 40, 40), count=2)
    _index(db, tmp_path / "lib", embedder)

    wrong_size = normalize(np.ones(999, dtype=np.float32))
    db.save_concept("bad", wrong_size, 0.5, 1)

    with pytest.raises(ValueError, match="dimension"):
        apply_concept(db, "bad")


def test_search_on_empty_index_returns_nothing(db, tmp_path, make_images, embedder):
    make_images(tmp_path / "q", (200, 40, 40), count=1)
    assert search_mod.by_examples(db, tmp_path / "q", embedder, limit=5) == []


def test_video_frames_collapse_to_one_hit(db, tmp_path, make_images, embedder):
    """Several frames of one video must not appear as several results."""
    make_images(tmp_path / "lib", (200, 40, 40), count=1)
    _index(db, tmp_path / "lib", embedder)

    file_id = db.conn.execute("SELECT id FROM files").fetchone()["id"]
    _, vectors = embedder.embed_paths([tmp_path / "lib" / "img00.png"])
    db.add_embeddings(file_id, np.vstack([vectors[0]] * 4), [0.0, 1.0, 2.0, 3.0])
    db.commit()

    make_images(tmp_path / "q", (200, 40, 40), count=1)
    hits = search_mod.by_examples(db, tmp_path / "q", embedder, limit=10)
    assert len(hits) == 1


def test_rematch_without_people_is_a_noop(db):
    assert rematch_faces(db) == 0


def test_rematch_assigns_faces_to_a_new_person(db, tmp_path):
    """Registering someone later must find them without re-embedding the library."""
    rng = np.random.default_rng(7)
    face = normalize(rng.standard_normal(16).astype(np.float32))

    file_id = db.upsert_file(tmp_path / "a.jpg", "image", 1, 1)
    db.add_file_faces(file_id, [{"vector": face, "person_id": None, "score": None}])
    db.commit()

    person_id = db.upsert_person("Sam")
    db.replace_person_faces(person_id, face, ["ref.jpg"])

    assert rematch_faces(db) == 1
    assert [r["path"] for r in db.files_for_person("Sam")] == [str(tmp_path / "a.jpg")]


def test_rematch_leaves_strangers_unassigned(db, tmp_path):
    rng = np.random.default_rng(11)
    stranger = normalize(rng.standard_normal(16).astype(np.float32))
    reference = normalize(-stranger)  # maximally dissimilar

    file_id = db.upsert_file(tmp_path / "a.jpg", "image", 1, 1)
    db.add_file_faces(file_id, [{"vector": stranger, "person_id": None, "score": None}])
    db.commit()

    person_id = db.upsert_person("Sam")
    db.replace_person_faces(person_id, reference, ["ref.jpg"])

    assert rematch_faces(db) == 0
