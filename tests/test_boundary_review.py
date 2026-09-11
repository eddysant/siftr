"""Reviewing a tag's boundary.

Counter-examples are the highest-leverage input a tag can take — measured, five
took recall from 5/8 to 7/8 where doubling the positives did nothing. These cover
surfacing the files worth asking about, and making an answer stick.
"""

import pytest

from siftr.index import build_index
from siftr.service import boundary_files, relearn, review_boundary_file, teach_from_paths


@pytest.fixture
def library(db, tmp_path, make_images, embedder):
    """A graded library plus a tag taught from its reddest few."""
    for i in range(12):
        make_images(tmp_path / "lib" / f"g{i}", (240 - i * 16, 40 + i * 10, 60), count=1)
    build_index(db, tmp_path / "lib", embedder, detect_faces=False)
    examples = sorted((tmp_path / "lib").rglob("*.png"))[:3]
    teach_from_paths(db, "reds", examples, embedder)
    return db, tmp_path, examples


# ------------------------------------------------------------------ surfacing


def test_boundary_returns_files_nearest_the_threshold(library, embedder):
    db, _tmp, _ex = library
    files = boundary_files(db, "reds", limit=5)

    assert len(files) == 5
    distances = [f["distance"] for f in files]
    assert distances == sorted(distances), "closest to the line first"


def test_boundary_spans_both_sides_of_the_line(library):
    db, _tmp, _ex = library
    files = boundary_files(db, "reds", limit=10)
    sides = {f["matching"] for f in files}
    assert sides == {True, False}, "a boundary with only one side teaches nothing"


def test_boundary_reports_score_and_side(library):
    db, _tmp, _ex = library
    row = boundary_files(db, "reds", limit=1)[0]
    assert set(row) >= {"path", "name", "kind", "score", "matching", "distance"}
    assert row["distance"] == pytest.approx(abs(row["score"] - db.get_concept("reds")["threshold"]))


def test_boundary_of_an_unknown_tag_raises(db):
    with pytest.raises(KeyError):
        boundary_files(db, "ghost")


def test_boundary_limit_is_respected(library):
    db, _tmp, _ex = library
    assert len(boundary_files(db, "reds", limit=2)) == 2


# -------------------------------------------------------------------- review


def test_rejecting_a_file_removes_the_tag_from_it(library, embedder):
    """The guaranteed half of a rejection.

    Its effect on the *threshold* is clamped — one counter-example must not be
    able to exclude real matches — so the cutoff may not move. Taking the tag off
    the file the user rejected is not negotiable.
    """
    db, _tmp, _ex = library
    above = [f for f in boundary_files(db, "reds", limit=12) if f["matching"]]
    target = above[-1]["path"]

    review_boundary_file(db, "reds", target, False, embedder)

    file_id = db.file_id_for_path(target)
    assert "reds" not in db.effective_tags().get(file_id, [])
    assert db.rejections(int(db.get_concept("reds")["id"])) == [target]


def test_a_rejection_is_fed_back_as_a_counter_example(db, tmp_path, make_images, embedder):
    """Recorded is not enough: a stored rejection has to reach the next teach.

    Measured against the same file passed explicitly as a negative, on a fixture
    where that demonstrably moves the threshold — comparing on one where it does
    not would pass whether or not the rejection was used at all.
    """
    for i in range(12):
        make_images(tmp_path / "lib" / f"g{i:02d}", (240 - i * 16, 40 + i * 10, 60), count=1)
    build_index(db, tmp_path / "lib", embedder, detect_faces=False)
    files = sorted((tmp_path / "lib").rglob("*.png"))
    examples, reject = files[:3], files[7]

    plain = teach_from_paths(db, "baseline", examples, embedder)
    explicit = teach_from_paths(db, "explicit", examples, embedder, negatives=[reject])
    assert explicit.threshold != pytest.approx(plain.threshold), (
        "fixture is useless unless the negative actually moves the threshold"
    )

    teach_from_paths(db, "stored", examples, embedder)
    concept_id = int(db.get_concept("stored")["id"])
    db.add_rejection(concept_id, reject)
    reteach = teach_from_paths(db, "stored", examples, embedder)

    assert reteach.threshold == pytest.approx(explicit.threshold), (
        "a stored rejection must act exactly like an explicitly dropped negative"
    )


def test_confirming_a_file_pins_the_tag_on_it(library, embedder):
    db, _tmp, _ex = library
    below = [f for f in boundary_files(db, "reds", limit=12) if not f["matching"]]
    target = below[0]["path"]

    review_boundary_file(db, "reds", target, True, embedder)

    file_id = db.file_id_for_path(target)
    assert "reds" in db.effective_tags().get(file_id, [])


def test_confirming_clears_a_previous_rejection(library, embedder):
    db, _tmp, _ex = library
    concept_id = int(db.get_concept("reds")["id"])
    target = boundary_files(db, "reds", limit=12)[0]["path"]

    review_boundary_file(db, "reds", target, False, embedder)
    assert target in db.rejections(concept_id)

    review_boundary_file(db, "reds", target, True, embedder)
    assert target not in db.rejections(concept_id), "changing your mind must work"


def test_reviewing_an_unknown_tag_raises(db, tmp_path, embedder):
    with pytest.raises(KeyError):
        review_boundary_file(db, "ghost", tmp_path / "x.png", False, embedder)


def test_relearn_with_no_pins_keeps_the_prototype(library, embedder):
    db, _tmp, _ex = library
    before = db.get_concept("reds")["prototype"]
    relearn(db, "reds", embedder)
    assert db.get_concept("reds")["prototype"] == before


def test_rejections_cascade_when_a_tag_is_deleted(library, embedder):
    db, _tmp, _ex = library
    review_boundary_file(db, "reds", boundary_files(db, "reds", 1)[0]["path"], False, embedder)
    db.delete_concept("reds")
    assert db.conn.execute("SELECT count(*) FROM concept_rejections").fetchone()[0] == 0


def test_repeated_rejection_is_idempotent(library, embedder):
    db, _tmp, _ex = library
    concept_id = int(db.get_concept("reds")["id"])
    target = boundary_files(db, "reds", 1)[0]["path"]
    review_boundary_file(db, "reds", target, False, embedder)
    review_boundary_file(db, "reds", target, False, embedder)
    assert db.rejections(concept_id).count(target) == 1
