"""SQLite storage for the media index, learned concepts, and known people.

Schema notes
------------
* Embeddings live in ``BLOB`` columns as raw float32 (see :mod:`siftr.vectors`),
  not JSON. A 50k-file library holds hundreds of thousands of vectors; JSON text
  would roughly quadruple the database and dominate load time.
* ``files`` is keyed on the absolute path, and carries ``size``/``mtime_ns`` so a
  re-scan can skip unchanged files without re-embedding them.
* Tags are a real join table rather than a JSON column, so "every file tagged X"
  is an indexed lookup instead of a full scan plus JSON parsing.
* A file may contain several concepts and several people at once, so both
  relationships are many-to-many.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path

import numpy as np

from .vectors import from_blob, to_blob

SCHEMA_VERSION = 4

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS files (
    id        INTEGER PRIMARY KEY,
    path      TEXT NOT NULL UNIQUE,
    kind      TEXT NOT NULL,              -- 'image' | 'video'
    size      INTEGER NOT NULL,
    mtime_ns  INTEGER NOT NULL,
    -- Frames sampled when this file was indexed. Raising --video-samples must
    -- re-index videos that were embedded at the old, lower rate; without this
    -- they looked "unchanged" and silently kept their coarser coverage.
    samples   INTEGER NOT NULL DEFAULT 1,
    indexed_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- One row per embedded view of a file: a single row for an image, or one row
-- per sampled frame for a video (frame_time = seconds into the clip).
CREATE TABLE IF NOT EXISTS embeddings (
    id          INTEGER PRIMARY KEY,
    file_id     INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    frame_time  REAL NOT NULL DEFAULT 0.0,
    vector      BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_embeddings_file ON embeddings(file_id);

-- Folders the user has opened for indexing. Persisted because the API's
-- read allowlist is rebuilt from this at startup: a library indexed in an
-- earlier session (or from the CLI) must still be readable by the UI, and
-- consent to read a folder is exactly what indexing it expresses.
CREATE TABLE IF NOT EXISTS roots (
    path       TEXT PRIMARY KEY,
    added_at   TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS concepts (
    id         INTEGER PRIMARY KEY,
    name       TEXT NOT NULL UNIQUE,
    threshold  REAL NOT NULL,
    n_examples INTEGER NOT NULL DEFAULT 0,
    prototype  BLOB NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS file_concepts (
    file_id    INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    concept_id INTEGER NOT NULL REFERENCES concepts(id) ON DELETE CASCADE,
    score      REAL NOT NULL,
    PRIMARY KEY (file_id, concept_id)
);
CREATE INDEX IF NOT EXISTS idx_file_concepts_concept ON file_concepts(concept_id);

-- Manual corrections layered over the model's decisions. 'on' forces a tag
-- that scoring missed, 'off' suppresses one it got wrong. Kept separate from
-- file_concepts so a re-score can freely replace model output without
-- destroying the user's corrections.
CREATE TABLE IF NOT EXISTS concept_overrides (
    file_id    INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    concept_id INTEGER NOT NULL REFERENCES concepts(id) ON DELETE CASCADE,
    state      TEXT NOT NULL CHECK (state IN ('on', 'off')),
    PRIMARY KEY (file_id, concept_id)
);
CREATE INDEX IF NOT EXISTS idx_overrides_concept ON concept_overrides(concept_id);

CREATE TABLE IF NOT EXISTS people (
    id         INTEGER PRIMARY KEY,
    name       TEXT NOT NULL UNIQUE,
    n_examples INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Several reference embeddings per person: faces vary enough with age, pose and
-- lighting that a single averaged centroid loses real accuracy. Matching takes
-- the best similarity over all of a person's references.
CREATE TABLE IF NOT EXISTS person_faces (
    id        INTEGER PRIMARY KEY,
    person_id INTEGER NOT NULL REFERENCES people(id) ON DELETE CASCADE,
    vector    BLOB NOT NULL,
    source    TEXT
);
CREATE INDEX IF NOT EXISTS idx_person_faces_person ON person_faces(person_id);

-- Faces actually found in library files, with the person they were matched to
-- (NULL = detected but unidentified).
CREATE TABLE IF NOT EXISTS file_faces (
    id         INTEGER PRIMARY KEY,
    file_id    INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    person_id  INTEGER REFERENCES people(id) ON DELETE SET NULL,
    score      REAL,
    frame_time REAL NOT NULL DEFAULT 0.0,
    bbox       TEXT,
    vector     BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_file_faces_file ON file_faces(file_id);
CREATE INDEX IF NOT EXISTS idx_file_faces_person ON file_faces(person_id);
"""


class Database:
    """Thin, explicit wrapper over the siftr SQLite file.

    Usable as a context manager; ``row_factory`` is set to :class:`sqlite3.Row`
    so callers can address columns by name.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        # Foreign keys are off by default in SQLite; without this the ON DELETE
        # CASCADE clauses above would silently do nothing.
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.executescript(_SCHEMA)
        self.conn.execute(
            "INSERT INTO meta (key, value) VALUES ('schema_version', ?) "
            "ON CONFLICT(key) DO NOTHING",
            (str(SCHEMA_VERSION),),
        )
        self.conn.commit()

    def __enter__(self) -> Database:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()

    # ------------------------------------------------------------------ files

    def upsert_file(self, path: Path, kind: str, size: int, mtime_ns: int, samples: int = 1) -> int:
        """Record a file and return its id, replacing any prior embeddings.

        Re-indexing a changed file must not leave its old vectors behind, so the
        embedding rows are cleared here rather than at the call site.
        """
        cur = self.conn.execute(
            """
            INSERT INTO files (path, kind, size, mtime_ns, samples)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                kind = excluded.kind,
                size = excluded.size,
                mtime_ns = excluded.mtime_ns,
                samples = excluded.samples,
                indexed_at = datetime('now')
            RETURNING id
            """,
            (str(path), kind, size, mtime_ns, samples),
        )
        file_id = int(cur.fetchone()[0])
        self.conn.execute("DELETE FROM embeddings WHERE file_id = ?", (file_id,))
        self.conn.execute("DELETE FROM file_faces WHERE file_id = ?", (file_id,))
        # Deliberately NOT clearing concept_overrides: a re-index re-derives the
        # model's opinion, but the user's corrections are theirs to keep.
        self.conn.execute("DELETE FROM file_concepts WHERE file_id = ?", (file_id,))
        return file_id

    def is_unchanged(self, path: Path, size: int, mtime_ns: int, samples: int = 1) -> bool:
        """True if this exact file is already indexed at this sample rate.

        The sample check is what makes raising --video-samples take effect
        without --force: a video embedded from 4 frames is not "unchanged" when
        8 are now wanted. Files indexed at a *higher* rate are left alone, so
        lowering the setting does not throw away work.
        """
        row = self.conn.execute(
            """
            SELECT 1 FROM files f
            WHERE f.path = ? AND f.size = ? AND f.mtime_ns = ? AND f.samples >= ?
              AND EXISTS (SELECT 1 FROM embeddings e WHERE e.file_id = f.id)
            """,
            (str(path), size, mtime_ns, samples),
        ).fetchone()
        return row is not None

    def forget_missing(self) -> int:
        """Drop index rows for files that no longer exist on disk."""
        rows = self.conn.execute("SELECT id, path FROM files").fetchall()
        gone = [(r["id"],) for r in rows if not Path(r["path"]).exists()]
        if gone:
            self.conn.executemany("DELETE FROM files WHERE id = ?", gone)
            self.conn.commit()
        return len(gone)

    def count_files(self) -> int:
        return int(self.conn.execute("SELECT count(*) FROM files").fetchone()[0])

    def rename_file(self, old: Path, new: Path) -> bool:
        """Point an indexed file at its new path after a rename on disk.

        The index keys on absolute path, so a rename that skipped this would
        orphan every embedding, face and tag belonging to the file.
        """
        cur = self.conn.execute("UPDATE files SET path = ? WHERE path = ?", (str(new), str(old)))
        return cur.rowcount > 0

    def path_for(self, file_id: int) -> str | None:
        row = self.conn.execute("SELECT path FROM files WHERE id = ?", (file_id,)).fetchone()
        return row["path"] if row else None

    # ------------------------------------------------------------- embeddings

    def add_embeddings(
        self, file_id: int, vectors: np.ndarray, frame_times: Iterable[float] | None = None
    ) -> None:
        stack = np.atleast_2d(vectors)
        times = list(frame_times) if frame_times is not None else [0.0] * len(stack)
        self.conn.executemany(
            "INSERT INTO embeddings (file_id, frame_time, vector) VALUES (?, ?, ?)",
            [(file_id, float(t), to_blob(v)) for v, t in zip(stack, times, strict=True)],
        )

    def iter_embeddings(self, batch: int = 4096) -> Iterator[tuple[list[sqlite3.Row], np.ndarray]]:
        """Stream (rows, matrix) batches of every embedding in the library.

        Yielding batches keeps peak memory bounded: a library large enough to
        matter will not fit its whole embedding matrix comfortably in RAM.
        """
        cur = self.conn.execute(
            """
            SELECT e.id, e.file_id, e.frame_time, e.vector, f.path, f.kind
            FROM embeddings e JOIN files f ON f.id = e.file_id
            """
        )
        while True:
            rows = cur.fetchmany(batch)
            if not rows:
                return
            yield rows, np.vstack([from_blob(r["vector"]) for r in rows])

    # --------------------------------------------------------------- concepts

    def save_concept(
        self, name: str, prototype: np.ndarray, threshold: float, n_examples: int
    ) -> int:
        cur = self.conn.execute(
            """
            INSERT INTO concepts (name, threshold, n_examples, prototype)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                threshold = excluded.threshold,
                n_examples = excluded.n_examples,
                prototype = excluded.prototype
            RETURNING id
            """,
            (name, float(threshold), int(n_examples), to_blob(prototype)),
        )
        concept_id = int(cur.fetchone()[0])
        self.conn.commit()
        return concept_id

    def get_concept(self, name: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM concepts WHERE name = ?", (name,)).fetchone()

    def list_concepts(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            """
            SELECT c.*, (SELECT count(*) FROM file_concepts fc WHERE fc.concept_id = c.id)
                        AS n_matches
            FROM concepts c ORDER BY c.name
            """
        ).fetchall()

    def delete_concept(self, name: str) -> bool:
        cur = self.conn.execute("DELETE FROM concepts WHERE name = ?", (name,))
        self.conn.commit()
        return cur.rowcount > 0

    def set_file_concepts(self, concept_id: int, scored: Iterable[tuple[int, float]]) -> None:
        """Replace the stored matches for one concept."""
        self.conn.execute("DELETE FROM file_concepts WHERE concept_id = ?", (concept_id,))
        self.conn.executemany(
            "INSERT INTO file_concepts (file_id, concept_id, score) VALUES (?, ?, ?) "
            "ON CONFLICT(file_id, concept_id) DO UPDATE SET score = excluded.score",
            [(fid, concept_id, float(s)) for fid, s in scored],
        )
        self.conn.commit()

    def files_for_concept(self, name: str, limit: int | None = None) -> list[sqlite3.Row]:
        sql = """
            SELECT f.path, f.kind, fc.score
            FROM file_concepts fc
            JOIN files f ON f.id = fc.file_id
            JOIN concepts c ON c.id = fc.concept_id
            WHERE c.name = ?
            ORDER BY fc.score DESC
        """
        params: list = [name]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        return self.conn.execute(sql, params).fetchall()

    # ----------------------------------------------------------------- people

    def upsert_person(self, name: str) -> int:
        cur = self.conn.execute(
            "INSERT INTO people (name) VALUES (?) "
            "ON CONFLICT(name) DO UPDATE SET name = excluded.name RETURNING id",
            (name,),
        )
        return int(cur.fetchone()[0])

    def replace_person_faces(
        self, person_id: int, vectors: np.ndarray, sources: Iterable[str]
    ) -> None:
        stack = np.atleast_2d(vectors)
        self.conn.execute("DELETE FROM person_faces WHERE person_id = ?", (person_id,))
        self.conn.executemany(
            "INSERT INTO person_faces (person_id, vector, source) VALUES (?, ?, ?)",
            [(person_id, to_blob(v), s) for v, s in zip(stack, sources, strict=True)],
        )
        self.conn.execute("UPDATE people SET n_examples = ? WHERE id = ?", (len(stack), person_id))
        self.conn.commit()

    def all_person_references(self) -> tuple[list[tuple[int, str]], np.ndarray]:
        """Every registered reference face as (owner list, matrix) for matching."""
        rows = self.conn.execute(
            """
            SELECT pf.vector, p.id AS person_id, p.name
            FROM person_faces pf JOIN people p ON p.id = pf.person_id
            """
        ).fetchall()
        if not rows:
            return [], np.empty((0, 0), dtype=np.float32)
        owners = [(int(r["person_id"]), r["name"]) for r in rows]
        return owners, np.vstack([from_blob(r["vector"]) for r in rows])

    def person_reference_vectors(self, person_id: int) -> np.ndarray:
        rows = self.conn.execute(
            "SELECT vector FROM person_faces WHERE person_id = ?", (person_id,)
        ).fetchall()
        if not rows:
            return np.empty((0, 0), dtype=np.float32)
        return np.vstack([from_blob(r["vector"]) for r in rows])

    def person_reference_sources(self, person_id: int) -> list[str]:
        return [
            r["source"] or ""
            for r in self.conn.execute(
                "SELECT source FROM person_faces WHERE person_id = ?", (person_id,)
            )
        ]

    def face_vectors(self, face_ids: Sequence[int]) -> np.ndarray:
        """Embeddings for specific detected faces, used when naming a cluster."""
        if not face_ids:
            return np.empty((0, 0), dtype=np.float32)
        placeholders = ",".join("?" * len(face_ids))
        rows = self.conn.execute(
            f"SELECT vector FROM file_faces WHERE id IN ({placeholders})",
            [int(f) for f in face_ids],
        ).fetchall()
        if not rows:
            return np.empty((0, 0), dtype=np.float32)
        return np.vstack([from_blob(r["vector"]) for r in rows])

    def list_people(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            """
            SELECT p.*, (SELECT count(DISTINCT ff.file_id) FROM file_faces ff
                         WHERE ff.person_id = p.id) AS n_files
            FROM people p ORDER BY p.name
            """
        ).fetchall()

    def delete_person(self, name: str) -> bool:
        cur = self.conn.execute("DELETE FROM people WHERE name = ?", (name,))
        self.conn.commit()
        return cur.rowcount > 0

    def add_file_faces(self, file_id: int, faces: Iterable[dict]) -> None:
        self.conn.executemany(
            """
            INSERT INTO file_faces (file_id, person_id, score, frame_time, bbox, vector)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    file_id,
                    f.get("person_id"),
                    f.get("score"),
                    float(f.get("frame_time", 0.0)),
                    f.get("bbox"),
                    to_blob(f["vector"]),
                )
                for f in faces
            ],
        )

    def iter_unassigned_faces(self) -> Iterator[tuple[list[sqlite3.Row], np.ndarray]]:
        """Detected faces with no person yet — used to re-match after new people."""
        rows = self.conn.execute(
            "SELECT id, file_id, vector FROM file_faces WHERE person_id IS NULL"
        ).fetchall()
        if rows:
            yield rows, np.vstack([from_blob(r["vector"]) for r in rows])

    def assign_face(self, face_id: int, person_id: int, score: float) -> None:
        self.conn.execute(
            "UPDATE file_faces SET person_id = ?, score = ? WHERE id = ?",
            (person_id, float(score), face_id),
        )

    def files_for_person(self, name: str, limit: int | None = None) -> list[sqlite3.Row]:
        sql = """
            SELECT f.path, f.kind, max(ff.score) AS score
            FROM file_faces ff
            JOIN files f ON f.id = ff.file_id
            JOIN people p ON p.id = ff.person_id
            WHERE p.name = ?
            GROUP BY f.id
            ORDER BY score DESC
        """
        params: list = [name]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        return self.conn.execute(sql, params).fetchall()

    # ----------------------------------------------------------- multi-tag views

    def tags_by_file(self) -> dict[int, list[str]]:
        """Every scored file mapped to all the concept names it matched.

        One query rather than per-file lookups: the grid needs this for the whole
        library at once, and N queries over tens of thousands of files is the
        difference between instant and unusable.
        """
        rows = self.conn.execute(
            """
            SELECT fc.file_id, c.name
            FROM file_concepts fc JOIN concepts c ON c.id = fc.concept_id
            ORDER BY c.name
            """
        ).fetchall()
        out: dict[int, list[str]] = {}
        for row in rows:
            out.setdefault(int(row["file_id"]), []).append(row["name"])
        return out

    def library_view(self) -> list[sqlite3.Row]:
        """Every indexed file with its matched tags and identified people.

        ``tags`` and ``people`` come back as comma-separated strings; SQLite has
        no array type and group_concat keeps this to a single query.
        """
        return self.conn.execute(
            """
            SELECT
                f.id, f.path, f.kind, f.size, f.mtime_ns,
                (SELECT group_concat(c.name, ',')
                   FROM file_concepts fc JOIN concepts c ON c.id = fc.concept_id
                  WHERE fc.file_id = f.id) AS tags,
                (SELECT group_concat(DISTINCT p.name)
                   FROM file_faces ff JOIN people p ON p.id = ff.person_id
                  WHERE ff.file_id = f.id) AS people
            FROM files f
            ORDER BY f.path
            """
        ).fetchall()

    def files_matching_tags(
        self, names: Sequence[str], mode: str = "any", limit: int | None = None
    ) -> list[sqlite3.Row]:
        """Files matching several tags at once.

        ``mode="any"`` is the union, ``mode="all"`` the intersection. ALL is done
        with a HAVING count over the distinct matched concepts rather than
        chained joins, so it stays one query for any number of tags.
        """
        if not names:
            return []
        if mode not in {"any", "all"}:
            raise ValueError(f"mode must be 'any' or 'all', got {mode!r}")

        placeholders = ",".join("?" * len(names))
        sql = f"""
            SELECT f.id, f.path, f.kind,
                   max(fc.score) AS score,
                   count(DISTINCT c.id) AS matched,
                   group_concat(c.name, ',') AS tags
            FROM file_concepts fc
            JOIN concepts c ON c.id = fc.concept_id
            JOIN files f ON f.id = fc.file_id
            WHERE c.name IN ({placeholders})
            GROUP BY f.id
        """
        params: list = list(names)
        if mode == "all":
            sql += " HAVING count(DISTINCT c.id) = ?"
            params.append(len(set(names)))
        sql += " ORDER BY score DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        return self.conn.execute(sql, params).fetchall()

    # -------------------------------------------------------------- overrides

    def set_override(self, file_id: int, concept_id: int, state: str | None) -> None:
        """Force a tag on ('on'), suppress it ('off'), or defer to the model (None)."""
        if state is None:
            self.conn.execute(
                "DELETE FROM concept_overrides WHERE file_id = ? AND concept_id = ?",
                (file_id, concept_id),
            )
        else:
            if state not in {"on", "off"}:
                raise ValueError(f"state must be 'on', 'off' or None, got {state!r}")
            self.conn.execute(
                """
                INSERT INTO concept_overrides (file_id, concept_id, state)
                VALUES (?, ?, ?)
                ON CONFLICT(file_id, concept_id) DO UPDATE SET state = excluded.state
                """,
                (file_id, concept_id, state),
            )
        self.conn.commit()

    def effective_tags(self) -> dict[int, list[str]]:
        """Every file's final tag set: model matches, with overrides applied.

        This is the single source of truth for what a file is tagged with —
        what the grid shows and what gets written into filenames.
        """
        rows = self.conn.execute(
            """
            SELECT file_id, name FROM (
                SELECT fc.file_id AS file_id, c.name AS name
                  FROM file_concepts fc
                  JOIN concepts c ON c.id = fc.concept_id
                 WHERE NOT EXISTS (
                    SELECT 1 FROM concept_overrides o
                     WHERE o.file_id = fc.file_id
                       AND o.concept_id = fc.concept_id
                       AND o.state = 'off'
                 )
                UNION
                SELECT o.file_id AS file_id, c.name AS name
                  FROM concept_overrides o
                  JOIN concepts c ON c.id = o.concept_id
                 WHERE o.state = 'on'
            )
            ORDER BY name
            """
        ).fetchall()
        out: dict[int, list[str]] = {}
        for row in rows:
            out.setdefault(int(row["file_id"]), []).append(row["name"])
        return out

    def overrides_for_concept(self, concept_id: int, state: str) -> list[str]:
        """Paths of files a user has manually forced on/off for a tag.

        The 'on' set doubles as extra training examples: a correction is the
        strongest signal there is about what a tag should mean.
        """
        rows = self.conn.execute(
            """
            SELECT f.path FROM concept_overrides o
            JOIN files f ON f.id = o.file_id
            WHERE o.concept_id = ? AND o.state = ?
            """,
            (concept_id, state),
        ).fetchall()
        return [r["path"] for r in rows]

    def file_id_for_path(self, path) -> int | None:
        row = self.conn.execute("SELECT id FROM files WHERE path = ?", (str(path),)).fetchone()
        return int(row["id"]) if row else None

    # ------------------------------------------------------------------ roots

    def add_root(self, path) -> None:
        self.conn.execute(
            "INSERT INTO roots (path) VALUES (?) ON CONFLICT(path) DO NOTHING",
            (str(Path(path).expanduser().resolve()),),
        )
        self.conn.commit()

    def list_roots(self) -> list[str]:
        return [r["path"] for r in self.conn.execute("SELECT path FROM roots ORDER BY added_at")]

    def commit(self) -> None:
        self.conn.commit()
