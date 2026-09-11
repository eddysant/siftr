from pathlib import Path

import numpy as np
import pytest

from siftr.db import Database
from siftr.vectors import normalize


def _vec(seed, dim=16):
    rng = np.random.default_rng(seed)
    return normalize(rng.standard_normal(dim).astype(np.float32))


def test_upsert_file_is_idempotent_on_path(db, tmp_path):
    path = tmp_path / "a.jpg"
    first = db.upsert_file(path, "image", 100, 1)
    second = db.upsert_file(path, "image", 100, 1)
    assert first == second
    assert db.count_files() == 1


def test_reindexing_replaces_old_embeddings(db, tmp_path):
    """A changed file must not keep its stale vectors alongside the new ones."""
    path = tmp_path / "a.jpg"
    file_id = db.upsert_file(path, "image", 100, 1)
    db.add_embeddings(file_id, np.vstack([_vec(1), _vec(2)]), [0.0, 1.0])
    db.commit()

    file_id = db.upsert_file(path, "image", 200, 2)
    db.add_embeddings(file_id, _vec(3))
    db.commit()

    total = db.conn.execute("SELECT count(*) FROM embeddings").fetchone()[0]
    assert total == 1


def test_is_unchanged_requires_embeddings(db, tmp_path):
    """A file row with no vectors is not 'done' — an interrupted scan must resume."""
    path = tmp_path / "a.jpg"
    db.upsert_file(path, "image", 100, 1)
    db.commit()
    assert not db.is_unchanged(path, 100, 1)

    file_id = db.conn.execute("SELECT id FROM files WHERE path = ?", (str(path),)).fetchone()[0]
    db.add_embeddings(file_id, _vec(1))
    db.commit()
    assert db.is_unchanged(path, 100, 1)


def test_is_unchanged_false_when_mtime_moves(db, tmp_path):
    path = tmp_path / "a.jpg"
    file_id = db.upsert_file(path, "image", 100, 1)
    db.add_embeddings(file_id, _vec(1))
    db.commit()
    assert not db.is_unchanged(path, 100, 999)


def test_deleting_file_cascades_to_embeddings(db, tmp_path):
    """Guards the PRAGMA foreign_keys=ON that makes ON DELETE CASCADE real."""
    file_id = db.upsert_file(tmp_path / "a.jpg", "image", 1, 1)
    db.add_embeddings(file_id, _vec(1))
    db.commit()
    db.conn.execute("DELETE FROM files WHERE id = ?", (file_id,))
    db.commit()
    assert db.conn.execute("SELECT count(*) FROM embeddings").fetchone()[0] == 0


def test_forget_missing_drops_only_absent_files(db, tmp_path):
    present = tmp_path / "present.jpg"
    present.write_bytes(b"x")
    db.upsert_file(present, "image", 1, 1)
    db.upsert_file(tmp_path / "gone.jpg", "image", 1, 1)
    db.commit()

    assert db.forget_missing() == 1
    remaining = [r["path"] for r in db.conn.execute("SELECT path FROM files")]
    assert remaining == [str(present)]


def test_iter_embeddings_batches_cover_everything(db, tmp_path):
    for i in range(10):
        file_id = db.upsert_file(tmp_path / f"{i}.jpg", "image", 1, i)
        db.add_embeddings(file_id, _vec(i))
    db.commit()

    seen = sum(len(rows) for rows, _ in db.iter_embeddings(batch=3))
    assert seen == 10


def test_concept_save_is_an_upsert(db):
    first = db.save_concept("sunsets", _vec(1), 0.7, 5)
    second = db.save_concept("sunsets", _vec(2), 0.8, 9)
    assert first == second
    row = db.get_concept("sunsets")
    assert row["threshold"] == 0.8
    assert row["n_examples"] == 9


def test_set_file_concepts_replaces_prior_matches(db, tmp_path):
    concept_id = db.save_concept("c", _vec(1), 0.5, 3)
    ids = [db.upsert_file(tmp_path / f"{i}.jpg", "image", 1, i) for i in range(3)]
    db.commit()

    db.set_file_concepts(concept_id, [(ids[0], 0.9), (ids[1], 0.8)])
    db.set_file_concepts(concept_id, [(ids[2], 0.7)])

    rows = db.files_for_concept("c")
    assert [r["path"] for r in rows] == [str(tmp_path / "2.jpg")]


def test_files_for_concept_orders_by_score_descending(db, tmp_path):
    concept_id = db.save_concept("c", _vec(1), 0.5, 3)
    low = db.upsert_file(tmp_path / "low.jpg", "image", 1, 1)
    high = db.upsert_file(tmp_path / "high.jpg", "image", 1, 2)
    db.commit()
    db.set_file_concepts(concept_id, [(low, 0.6), (high, 0.95)])

    names = [Path(r["path"]).name for r in db.files_for_concept("c")]
    assert names == ["high.jpg", "low.jpg"]


def test_replace_person_faces_updates_count(db):
    person_id = db.upsert_person("Sam")
    db.replace_person_faces(person_id, np.vstack([_vec(1), _vec(2), _vec(3)]), ["a", "b", "c"])
    assert db.list_people()[0]["n_examples"] == 3

    db.replace_person_faces(person_id, _vec(4), ["d"])
    assert db.list_people()[0]["n_examples"] == 1
    owners, matrix = db.all_person_references()
    assert len(owners) == 1 and matrix.shape[0] == 1


def test_all_person_references_empty_is_safe(db):
    owners, matrix = db.all_person_references()
    assert owners == [] and matrix.size == 0


def test_files_for_person_takes_best_score_per_file(db, tmp_path):
    person_id = db.upsert_person("Sam")
    file_id = db.upsert_file(tmp_path / "a.mp4", "video", 1, 1)
    db.add_file_faces(
        file_id,
        [
            {"vector": _vec(1), "person_id": person_id, "score": 0.42, "frame_time": 0.0},
            {"vector": _vec(2), "person_id": person_id, "score": 0.88, "frame_time": 5.0},
        ],
    )
    db.commit()

    rows = db.files_for_person("Sam")
    assert len(rows) == 1
    assert rows[0]["score"] == pytest.approx(0.88)


def test_deleting_person_keeps_the_detected_faces(db, tmp_path):
    """ON DELETE SET NULL: forgetting a person must not delete indexing work."""
    person_id = db.upsert_person("Sam")
    file_id = db.upsert_file(tmp_path / "a.jpg", "image", 1, 1)
    db.add_file_faces(file_id, [{"vector": _vec(1), "person_id": person_id, "score": 0.9}])
    db.commit()

    assert db.delete_person("Sam")
    row = db.conn.execute("SELECT person_id FROM file_faces").fetchone()
    assert row is not None and row["person_id"] is None


def test_schema_survives_reopen(tmp_path):
    path = tmp_path / "index.db"
    with Database(path) as first:
        first.save_concept("c", _vec(1), 0.5, 2)
    with Database(path) as second:
        assert second.get_concept("c") is not None


# ----------------------------------------------------------------- migration


def test_an_old_index_gains_new_columns(tmp_path):
    """`CREATE TABLE IF NOT EXISTS` does nothing to an existing table, so every
    column added after v1 would simply be missing from an older index — failing
    with a bare "no such column" at the first query that touches it."""
    import sqlite3

    path = tmp_path / "old.db"
    # A v1-shaped files table: none of the later columns.
    old = sqlite3.connect(path)
    old.executescript(
        """
        CREATE TABLE files (
            id INTEGER PRIMARY KEY, path TEXT NOT NULL UNIQUE, kind TEXT NOT NULL,
            size INTEGER NOT NULL, mtime_ns INTEGER NOT NULL,
            indexed_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        INSERT INTO files (path, kind, size, mtime_ns) VALUES ('/a.jpg', 'image', 1, 1);
        """
    )
    old.commit()
    old.close()

    with Database(path) as db:
        columns = {r["name"] for r in db.conn.execute("PRAGMA table_info(files)")}
        assert {"samples", "content_hash", "phash", "pixels"} <= columns
        # And the existing row survives, with defaults filled in.
        row = db.conn.execute("SELECT path, samples FROM files").fetchone()
        assert row["path"] == "/a.jpg"
        assert row["samples"] == 1


def test_migration_is_idempotent(tmp_path):
    path = tmp_path / "i.db"
    with Database(path) as first:
        first.save_concept("c", _vec(1), 0.5, 2)
    for _ in range(3):
        with Database(path) as again:
            assert again.get_concept("c") is not None


def test_migrated_index_supports_the_new_queries(tmp_path):
    """The failure this prevents is at query time, not open time."""
    import sqlite3

    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.executescript(
        """
        CREATE TABLE concepts (
            id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE, threshold REAL NOT NULL,
            n_examples INTEGER NOT NULL DEFAULT 0, prototype BLOB NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        """
    )
    old.commit()
    old.close()

    with Database(path) as db:
        db.save_concept("t", _vec(1), 0.5, 2)
        assert db.set_verify_phrase("t", "a tattooed arm")
        assert db.get_concept("t")["verify_phrase"] == "a tattooed arm"
        assert db.set_destination("t", "/tmp/x")
        assert db.destinations() == {"t": "/tmp/x"}
