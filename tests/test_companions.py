"""Live Photo pairs, sidecars, and filing into folders.

These cover the rules that protect real photo libraries: a Live Photo is a still
plus a motion clip matched by stem, and renaming or moving one without the other
destroys the pairing silently.
"""

from pathlib import Path

import numpy as np
import pytest

from siftr.companions import (
    find_companions,
    group_paths,
    is_sidecar,
    tags_for_groups,
)
from siftr.db import Database
from siftr.naming import retag_name
from siftr.rename import apply_renames, plan_moves, plan_renames
from siftr.service import organize
from siftr.vectors import normalize


def touch(path: Path, content=b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _vec(seed, dim=16):
    rng = np.random.default_rng(seed)
    return normalize(rng.standard_normal(dim).astype(np.float32))


# ------------------------------------------------------------------- grouping


def test_live_photo_halves_group_together(tmp_path):
    still = tmp_path / "IMG_4821.HEIC"
    motion = tmp_path / "IMG_4821.MOV"
    groups = group_paths([still, motion])
    assert len(groups) == 1
    assert groups[0].primary == still, "the still owns the group"
    assert set(groups[0].members) == {still, motion}
    assert groups[0].is_paired


def test_sidecars_join_their_media_file(tmp_path):
    photo = tmp_path / "IMG_1.JPG"
    edits = tmp_path / "IMG_1.AAE"
    groups = group_paths([photo, edits])
    assert len(groups) == 1
    assert groups[0].primary == photo
    assert groups[0].sidecars() == [edits]


def test_stems_are_matched_case_insensitively(tmp_path):
    """APFS treats IMG_1.HEIC and img_1.mov as the same stem; so must we."""
    groups = group_paths([tmp_path / "IMG_1.HEIC", tmp_path / "img_1.mov"])
    assert len(groups) == 1


def test_same_stem_in_different_directories_does_not_group(tmp_path):
    groups = group_paths([tmp_path / "a" / "IMG_1.jpg", tmp_path / "b" / "IMG_1.mov"])
    assert len(groups) == 2


def test_unrelated_files_stay_separate(tmp_path):
    groups = group_paths([tmp_path / "a.jpg", tmp_path / "b.jpg"])
    assert len(groups) == 2


def test_a_bucket_of_only_sidecars_is_ignored(tmp_path):
    """No media file owns them, so inventing a primary would rename junk."""
    assert group_paths([tmp_path / "orphan.xmp", tmp_path / "orphan.aae"]) == []


def test_still_wins_primary_over_video(tmp_path):
    groups = group_paths([tmp_path / "x.MOV", tmp_path / "x.HEIC"])
    assert groups[0].primary.suffix == ".HEIC"


def test_is_sidecar_recognises_apple_and_adobe(tmp_path):
    assert is_sidecar(Path("a.AAE")) and is_sidecar(Path("a.xmp"))
    assert not is_sidecar(Path("a.jpg"))


def test_find_companions_reads_the_directory(tmp_path):
    touch(tmp_path / "IMG_9.HEIC")
    touch(tmp_path / "IMG_9.MOV")
    touch(tmp_path / "IMG_9.AAE")
    touch(tmp_path / "OTHER.JPG")
    found = find_companions(tmp_path / "IMG_9.HEIC")
    assert {p.name for p in found} == {"IMG_9.HEIC", "IMG_9.MOV", "IMG_9.AAE"}


def test_find_companions_on_a_lone_file(tmp_path):
    touch(tmp_path / "solo.jpg")
    assert [p.name for p in find_companions(tmp_path / "solo.jpg")] == ["solo.jpg"]


# ------------------------------------------------------------- shared tagging


def test_group_members_inherit_the_primarys_tags(tmp_path):
    still = tmp_path / "IMG_1.HEIC"
    motion = tmp_path / "IMG_1.MOV"
    groups = group_paths([still, motion])
    # The two halves matched different tags on their own — the bug this fixes.
    resolved = tags_for_groups(groups, {still: ["beach"], motion: ["beach", "surf"]})
    assert resolved[still] == ["beach"]
    assert resolved[motion] == ["beach"], "the clip must not keep its own tags"


def test_live_photo_stems_stay_identical_after_retagging(tmp_path):
    """The regression itself: different tags used to give different stems."""
    still, motion = tmp_path / "IMG_1.HEIC", tmp_path / "IMG_1.MOV"
    groups = group_paths([still, motion])
    resolved = tags_for_groups(groups, {still: ["beach"], motion: ["beach", "surf"]})
    known = ["beach", "surf"]

    new_still = retag_name(still.name, resolved[still], known)
    new_motion = retag_name(motion.name, resolved[motion], known)

    assert Path(new_still).stem == Path(new_motion).stem


# ------------------------------------------------------ renaming a whole group


def test_renaming_moves_both_halves_together(tmp_path):
    still = touch(tmp_path / "IMG_1.HEIC")
    motion = touch(tmp_path / "IMG_1.MOV")
    groups = group_paths([still, motion])
    resolved = tags_for_groups(groups, {still: ["beach"], motion: ["beach", "surf"]})

    apply_renames(plan_renames(list(resolved.items()), ["beach", "surf"]))

    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == ["IMG_1 [beach].HEIC", "IMG_1 [beach].MOV"]


def test_sidecar_follows_its_photo(tmp_path):
    photo = touch(tmp_path / "IMG_2.JPG")
    edits = touch(tmp_path / "IMG_2.AAE")
    groups = group_paths([photo, edits])
    resolved = tags_for_groups(groups, {photo: ["sunset"]})

    apply_renames(plan_renames(list(resolved.items()), ["sunset"]))

    assert (tmp_path / "IMG_2 [sunset].JPG").exists()
    assert (tmp_path / "IMG_2 [sunset].AAE").exists()
    assert not edits.exists()


# ---------------------------------------------------------------- move mode


def test_plan_moves_files_into_a_destination(tmp_path):
    plans, contested = plan_moves(
        [(tmp_path / "a.jpg", ["beach"])], {"beach": str(tmp_path / "Beach")}
    )
    assert len(plans) == 1
    assert plans[0].target == tmp_path / "Beach" / "a.jpg"
    assert contested == []


def test_tags_without_a_destination_do_not_move_files(tmp_path):
    plans, _ = plan_moves([(tmp_path / "a.jpg", ["untargeted"])], {"beach": "/somewhere"})
    assert plans == []


def test_highest_scoring_tag_wins_a_contested_file(tmp_path):
    """A file can match many tags but live in only one folder."""
    path = tmp_path / "a.jpg"
    plans, contested = plan_moves(
        [(path, ["beach", "surf"])],
        {"beach": str(tmp_path / "Beach"), "surf": str(tmp_path / "Surf")},
        scores={path: {"beach": 0.7, "surf": 0.95}},
    )
    assert plans[0].target.parent.name == "Surf"
    assert contested and "filed under surf" in contested[0]


def test_contested_ties_break_alphabetically(tmp_path):
    """Deterministic rather than dependent on dict ordering."""
    path = tmp_path / "a.jpg"
    plans, _ = plan_moves(
        [(path, ["zebra", "alpha"])],
        {"zebra": str(tmp_path / "Z"), "alpha": str(tmp_path / "A")},
        scores={path: {"zebra": 0.8, "alpha": 0.8}},
    )
    assert plans[0].target.parent.name == "A"


def test_already_filed_files_are_left_alone(tmp_path):
    destination = tmp_path / "Beach"
    photo = touch(destination / "a.jpg")
    plans, _ = plan_moves([(photo, ["beach"])], {"beach": str(destination)})
    assert plans == []


def test_move_creates_the_destination_and_relocates(tmp_path):
    photo = touch(tmp_path / "a.jpg", b"payload")
    plans, _ = plan_moves([(photo, ["beach"])], {"beach": str(tmp_path / "Beach")})
    result = apply_renames(plans)
    assert result.count == 1
    assert (tmp_path / "Beach" / "a.jpg").read_bytes() == b"payload"
    assert not photo.exists()


def test_move_takes_the_whole_live_photo_group(tmp_path):
    still = touch(tmp_path / "IMG_5.HEIC")
    motion = touch(tmp_path / "IMG_5.MOV")
    groups = group_paths([still, motion])
    resolved = tags_for_groups(groups, {still: ["beach"], motion: ["surf"]})

    plans, _ = plan_moves(list(resolved.items()), {"beach": str(tmp_path / "Beach")})
    apply_renames(plans)

    filed = sorted(p.name for p in (tmp_path / "Beach").iterdir())
    assert filed == ["IMG_5.HEIC", "IMG_5.MOV"], "the pair must not be split"


def test_move_does_not_overwrite_a_name_already_there(tmp_path):
    touch(tmp_path / "Beach" / "a.jpg", b"existing")
    photo = touch(tmp_path / "a.jpg", b"incoming")
    plans, _ = plan_moves([(photo, ["beach"])], {"beach": str(tmp_path / "Beach")})
    apply_renames(plans)
    assert (tmp_path / "Beach" / "a.jpg").read_bytes() == b"existing"
    assert (tmp_path / "Beach" / "a-2.jpg").read_bytes() == b"incoming"


# ------------------------------------------------------------ organize modes


def _library(tmp_path):
    db = Database(tmp_path / "i.db")
    concept_id = db.save_concept("beach", _vec(1), 0.5, 3)
    still = touch(tmp_path / "lib" / "IMG_1.HEIC")
    motion = touch(tmp_path / "lib" / "IMG_1.MOV")
    file_id = db.upsert_file(still, "image", 1, 1)
    db.upsert_file(motion, "video", 1, 1)
    db.commit()
    db.set_file_concepts(concept_id, [(file_id, 0.9)])
    return db, still, motion


def test_organize_off_touches_nothing(tmp_path):
    db, still, motion = _library(tmp_path)
    out = organize(db, mode="off")
    assert out["changed"] == 0
    assert still.exists() and motion.exists()
    db.close()


def test_organize_rename_keeps_the_pair_together(tmp_path):
    db, _still, _motion = _library(tmp_path)
    out = organize(db, mode="rename", root=tmp_path)
    assert out["changed"] == 2, "both halves renamed"
    names = sorted(p.name for p in (tmp_path / "lib").iterdir())
    assert names == ["IMG_1 [beach].HEIC", "IMG_1 [beach].MOV"]
    db.close()


def test_organize_move_files_the_pair(tmp_path):
    db, _still, _motion = _library(tmp_path)
    db.set_destination("beach", str(tmp_path / "Beach"))
    out = organize(db, mode="move", root=tmp_path)
    assert out["changed"] == 2
    assert sorted(p.name for p in (tmp_path / "Beach").iterdir()) == [
        "IMG_1.HEIC",
        "IMG_1.MOV",
    ]
    db.close()


def test_organize_move_without_destinations_explains_itself(tmp_path):
    db, _still, _motion = _library(tmp_path)
    out = organize(db, mode="move")
    assert out["changed"] == 0
    assert "no tag has a destination" in out["errors"][0]
    db.close()


def test_organize_rejects_an_unknown_mode(tmp_path):
    db, _s, _m = _library(tmp_path)
    with pytest.raises(ValueError):
        organize(db, mode="sideways")
    db.close()


def test_organize_dry_run_changes_nothing(tmp_path):
    db, still, _motion = _library(tmp_path)
    out = organize(db, mode="rename", dry_run=True)
    assert out["changed"] == 2
    assert still.exists(), "dry run must not touch the filesystem"
    db.close()


def test_organize_reports_how_many_groups_were_paired(tmp_path):
    db, _s, _m = _library(tmp_path)
    assert organize(db, mode="rename", dry_run=True)["paired"] == 1
    db.close()


def test_mode_setting_round_trips(tmp_path):
    db = Database(tmp_path / "i.db")
    assert db.get_setting("organize_mode", "rename") == "rename"
    db.set_setting("organize_mode", "move")
    assert db.get_setting("organize_mode") == "move"
    db.close()


def test_destination_survives_reteaching_a_tag(tmp_path):
    """Re-teaching replaces the prototype; it must not wipe the folder."""
    db = Database(tmp_path / "i.db")
    db.save_concept("beach", _vec(1), 0.5, 3)
    db.set_destination("beach", "/tmp/Beach")
    db.save_concept("beach", _vec(2), 0.6, 9)
    assert db.destinations() == {"beach": "/tmp/Beach"}
    db.close()


def test_clearing_a_destination(tmp_path):
    db = Database(tmp_path / "i.db")
    db.save_concept("beach", _vec(1), 0.5, 3)
    db.set_destination("beach", "/tmp/Beach")
    db.set_destination("beach", None)
    assert db.destinations() == {}
    db.close()
