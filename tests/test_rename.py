"""Renaming files on disk, and undoing it."""

import json

import pytest

from siftr.db import Database
from siftr.naming import retag_name
from siftr.rename import (
    MANIFEST_NAME,
    RenamePlan,
    apply_renames,
    plan_renames,
    read_manifest,
    undo_last,
)

KNOWN = ["glaze", "outdoor"]


def touch(path, content=b"x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


# ------------------------------------------------------------------- planning


def test_plan_only_includes_files_that_change(tmp_path):
    a = tmp_path / "a.jpg"
    b = tmp_path / "b [glaze].jpg"
    plans = plan_renames([(a, ["glaze"]), (b, ["glaze"])], KNOWN)
    assert [p.source.name for p in plans] == ["a.jpg"]


def test_plan_targets_are_correct(tmp_path):
    plans = plan_renames([(tmp_path / "a.jpg", ["outdoor", "glaze"])], KNOWN)
    assert plans[0].target.name == "a [glaze] [outdoor].jpg"


# ------------------------------------------------------------------- applying


def test_apply_renames_the_file(tmp_path):
    src = touch(tmp_path / "a.jpg")
    result = apply_renames(plan_renames([(src, ["glaze"])], KNOWN))
    assert result.count == 1
    assert not src.exists()
    assert (tmp_path / "a [glaze].jpg").exists()


def test_apply_is_idempotent_across_runs(tmp_path):
    src = touch(tmp_path / "a.jpg")
    apply_renames(plan_renames([(src, ["glaze"])], KNOWN))
    current = tmp_path / "a [glaze].jpg"
    second = apply_renames(plan_renames([(current, ["glaze"])], KNOWN))
    assert second.count == 0
    assert current.exists()
    assert list(tmp_path.glob("*.jpg")) == [current]


def test_apply_never_overwrites_an_existing_file(tmp_path):
    """Two files can legitimately want the same tagged name."""
    a = touch(tmp_path / "sub1" / "IMG.jpg", b"one")
    occupied = touch(tmp_path / "sub1" / "IMG [glaze].jpg", b"other")

    apply_renames(plan_renames([(a, ["glaze"])], KNOWN))

    assert occupied.read_bytes() == b"other"
    assert (tmp_path / "sub1" / "IMG [glaze]-2.jpg").read_bytes() == b"one"


def test_apply_resolves_collisions_within_one_batch(tmp_path):
    a = touch(tmp_path / "x.jpg", b"a")
    b = touch(tmp_path / "x [outdoor].jpg", b"b")
    # Both want to become "x [glaze].jpg" is impossible, but both wanting the
    # same target after retagging is: b loses its tag, a gains none.
    plans = [
        RenamePlan(a, tmp_path / "same.jpg"),
        RenamePlan(b, tmp_path / "same.jpg"),
    ]
    result = apply_renames(plans)
    assert result.count == 2
    names = sorted(p.name for p in tmp_path.glob("*.jpg"))
    assert names == ["same-2.jpg", "same.jpg"]


def test_apply_defers_a_rename_blocked_by_another(tmp_path):
    """a->b while b->c is pending must not destroy b."""
    a = touch(tmp_path / "a.jpg", b"A")
    b = touch(tmp_path / "b.jpg", b"B")
    plans = [RenamePlan(a, tmp_path / "b.jpg"), RenamePlan(b, tmp_path / "c.jpg")]

    result = apply_renames(plans)

    assert result.count == 2
    assert (tmp_path / "c.jpg").read_bytes() == b"B"
    assert (tmp_path / "b.jpg").read_bytes() == b"A"


def test_apply_skips_a_circular_dependency_rather_than_forcing_it(tmp_path):
    a = touch(tmp_path / "a.jpg", b"A")
    b = touch(tmp_path / "b.jpg", b"B")
    plans = [RenamePlan(a, tmp_path / "b.jpg"), RenamePlan(b, tmp_path / "a.jpg")]

    result = apply_renames(plans)

    assert result.count == 0
    assert result.skipped == 2
    assert a.read_bytes() == b"A" and b.read_bytes() == b"B"


def test_apply_skips_a_vanished_source(tmp_path):
    plans = [RenamePlan(tmp_path / "ghost.jpg", tmp_path / "ghost [glaze].jpg")]
    result = apply_renames(plans)
    assert result.count == 0 and result.skipped == 1


def test_dry_run_touches_nothing(tmp_path):
    src = touch(tmp_path / "a.jpg")
    result = apply_renames(plan_renames([(src, ["glaze"])], KNOWN), dry_run=True)
    assert result.count == 1
    assert src.exists()
    assert not (tmp_path / "a [glaze].jpg").exists()


def test_apply_does_not_clobber_a_broken_symlink(tmp_path):
    """exists() is False for a broken link, but the name is still taken."""
    src = touch(tmp_path / "a.jpg", b"real")
    link = tmp_path / "a [glaze].jpg"
    link.symlink_to(tmp_path / "nonexistent-target.jpg")

    apply_renames(plan_renames([(src, ["glaze"])], KNOWN))

    assert link.is_symlink()
    assert (tmp_path / "a [glaze]-2.jpg").read_bytes() == b"real"


# ---------------------------------------------------------- index consistency


def test_apply_updates_the_index_path(tmp_path):
    src = touch(tmp_path / "a.jpg")
    with Database(tmp_path / "i.db") as db:
        db.upsert_file(src, "image", 1, 1)
        db.commit()

        apply_renames(plan_renames([(src, ["glaze"])], KNOWN), db=db)

        paths = [r["path"] for r in db.conn.execute("SELECT path FROM files")]
        assert paths == [str(tmp_path / "a [glaze].jpg")]


def test_index_path_follows_the_collision_suffixed_target(tmp_path):
    """The DB must record where the file actually went, not where it wanted to."""
    src = touch(tmp_path / "IMG.jpg", b"one")
    touch(tmp_path / "IMG [glaze].jpg", b"other")

    with Database(tmp_path / "i.db") as db:
        db.upsert_file(src, "image", 1, 1)
        db.commit()
        apply_renames(plan_renames([(src, ["glaze"])], KNOWN), db=db)
        paths = [r["path"] for r in db.conn.execute("SELECT path FROM files")]
        assert paths == [str(tmp_path / "IMG [glaze]-2.jpg")]


# ------------------------------------------------------------------ undo


def test_manifest_records_the_batch(tmp_path):
    src = touch(tmp_path / "a.jpg")
    apply_renames(plan_renames([(src, ["glaze"])], KNOWN), root=tmp_path)

    history = read_manifest(tmp_path)
    assert len(history) == 1
    assert history[0]["renames"][0][1].endswith("a [glaze].jpg")


def test_undo_restores_the_original_name(tmp_path):
    src = touch(tmp_path / "a.jpg", b"payload")
    apply_renames(plan_renames([(src, ["glaze"])], KNOWN), root=tmp_path)

    result = undo_last(tmp_path)

    assert result.count == 1
    assert src.read_bytes() == b"payload"
    assert not (tmp_path / "a [glaze].jpg").exists()


def test_undo_pops_only_the_last_batch(tmp_path):
    a = touch(tmp_path / "a.jpg")
    apply_renames(plan_renames([(a, ["glaze"])], KNOWN), root=tmp_path)
    b = touch(tmp_path / "b.jpg")
    apply_renames(plan_renames([(b, ["outdoor"])], KNOWN), root=tmp_path)

    undo_last(tmp_path)

    assert (tmp_path / "b.jpg").exists()
    assert (tmp_path / "a [glaze].jpg").exists(), "earlier batch must survive"
    assert len(read_manifest(tmp_path)) == 1


def test_undo_removes_the_manifest_when_history_is_exhausted(tmp_path):
    src = touch(tmp_path / "a.jpg")
    apply_renames(plan_renames([(src, ["glaze"])], KNOWN), root=tmp_path)
    undo_last(tmp_path)
    assert not (tmp_path / MANIFEST_NAME).exists()


def test_undo_restores_index_paths_too(tmp_path):
    src = touch(tmp_path / "a.jpg")
    with Database(tmp_path / "i.db") as db:
        db.upsert_file(src, "image", 1, 1)
        db.commit()
        apply_renames(plan_renames([(src, ["glaze"])], KNOWN), db=db, root=tmp_path)
        undo_last(tmp_path, db=db)
        paths = [r["path"] for r in db.conn.execute("SELECT path FROM files")]
        assert paths == [str(src)]


def test_undo_with_no_history_is_a_noop(tmp_path):
    assert undo_last(tmp_path).count == 0


def test_corrupt_manifest_is_ignored_not_fatal(tmp_path):
    (tmp_path / MANIFEST_NAME).write_text("{ not json")
    assert read_manifest(tmp_path) == []
    assert undo_last(tmp_path).count == 0


def test_manifest_survives_many_batches(tmp_path):
    for i in range(5):
        touch(tmp_path / f"f{i}.jpg")
        apply_renames(plan_renames([(tmp_path / f"f{i}.jpg", ["glaze"])], KNOWN), root=tmp_path)
    assert len(read_manifest(tmp_path)) == 5
    assert json.loads((tmp_path / MANIFEST_NAME).read_text())


# ------------------------------------------------------- round trip with tags


@pytest.mark.parametrize(
    "start,tags",
    [
        ("IMG_1.jpg", ["glaze"]),
        ("IMG_1 [2].jpg", ["glaze", "outdoor"]),
        ("holiday photo.png", ["outdoor"]),
    ],
)
def test_rename_then_undo_is_a_perfect_round_trip(tmp_path, start, tags):
    src = touch(tmp_path / start, b"data")
    apply_renames(plan_renames([(src, tags)], KNOWN), root=tmp_path)
    assert not src.exists()

    undo_last(tmp_path)

    assert src.read_bytes() == b"data"
    assert sorted(p.name for p in tmp_path.glob("*.*") if p.name != MANIFEST_NAME) == [start]


def test_retag_and_rename_agree(tmp_path):
    """plan_renames must target exactly what retag_name computes."""
    src = touch(tmp_path / "a [2].jpg")
    expected = retag_name(src.name, ["glaze"], KNOWN)
    apply_renames(plan_renames([(src, ["glaze"])], KNOWN))
    assert (tmp_path / expected).exists()
