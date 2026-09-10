"""Manual corrections layered over the model's tag decisions."""

import numpy as np
import pytest

from siftr.vectors import normalize


def _vec(seed, dim=16):
    rng = np.random.default_rng(seed)
    return normalize(rng.standard_normal(dim).astype(np.float32))


@pytest.fixture
def scored(db, tmp_path):
    """A library where the model tagged file 0 with 'glaze' and nothing else."""
    concept_id = db.save_concept("glaze", _vec(1), 0.5, 3)
    ids = [db.upsert_file(tmp_path / f"{i}.jpg", "image", 1, i) for i in range(3)]
    db.commit()
    db.set_file_concepts(concept_id, [(ids[0], 0.9)])
    return db, concept_id, ids


def test_effective_tags_match_the_model_with_no_overrides(scored):
    db, _cid, ids = scored
    assert db.effective_tags() == {ids[0]: ["glaze"]}


def test_override_on_forces_a_tag_the_model_missed(scored):
    db, cid, ids = scored
    db.set_override(ids[1], cid, "on")
    tags = db.effective_tags()
    assert tags[ids[1]] == ["glaze"]
    assert tags[ids[0]] == ["glaze"]


def test_override_off_suppresses_a_wrong_tag(scored):
    db, cid, ids = scored
    db.set_override(ids[0], cid, "off")
    assert ids[0] not in db.effective_tags()


def test_clearing_an_override_returns_to_the_model(scored):
    db, cid, ids = scored
    db.set_override(ids[0], cid, "off")
    assert ids[0] not in db.effective_tags()
    db.set_override(ids[0], cid, None)
    assert db.effective_tags()[ids[0]] == ["glaze"]


def test_rescoring_does_not_destroy_overrides(scored):
    """The model's opinion is replaceable; the user's correction is not."""
    db, cid, ids = scored
    db.set_override(ids[2], cid, "on")
    db.set_file_concepts(cid, [(ids[1], 0.8)])  # a fresh scoring pass
    tags = db.effective_tags()
    assert tags[ids[2]] == ["glaze"], "manual 'on' must survive re-scoring"
    assert tags[ids[1]] == ["glaze"]
    assert ids[0] not in tags


def test_reindexing_a_file_does_not_destroy_its_overrides(scored, tmp_path):
    db, cid, ids = scored
    db.set_override(ids[0], cid, "off")
    db.upsert_file(tmp_path / "0.jpg", "image", 999, 999)  # file changed on disk
    db.commit()
    assert ids[0] not in db.effective_tags()


def test_invalid_override_state_is_rejected(scored):
    db, cid, ids = scored
    with pytest.raises(ValueError):
        db.set_override(ids[0], cid, "maybe")


def test_overrides_for_concept_lists_corrections(scored, tmp_path):
    db, cid, ids = scored
    db.set_override(ids[1], cid, "on")
    db.set_override(ids[2], cid, "off")
    assert db.overrides_for_concept(cid, "on") == [str(tmp_path / "1.jpg")]
    assert db.overrides_for_concept(cid, "off") == [str(tmp_path / "2.jpg")]


def test_deleting_a_concept_removes_its_overrides(scored):
    db, cid, ids = scored
    db.set_override(ids[1], cid, "on")
    db.delete_concept("glaze")
    assert db.conn.execute("SELECT count(*) FROM concept_overrides").fetchone()[0] == 0


def test_file_id_for_path_roundtrips(scored, tmp_path):
    db, _cid, ids = scored
    assert db.file_id_for_path(tmp_path / "0.jpg") == ids[0]
    assert db.file_id_for_path(tmp_path / "nope.jpg") is None


# ---------------------------------------------------------------- multi-tag


@pytest.fixture
def two_tags(db, tmp_path):
    glaze = db.save_concept("glaze", _vec(1), 0.5, 3)
    outdoor = db.save_concept("outdoor", _vec(2), 0.5, 3)
    ids = [db.upsert_file(tmp_path / f"{i}.jpg", "image", 1, i) for i in range(3)]
    db.commit()
    db.set_file_concepts(glaze, [(ids[0], 0.9), (ids[1], 0.7)])
    db.set_file_concepts(outdoor, [(ids[1], 0.8), (ids[2], 0.6)])
    return db, ids


def test_any_mode_is_the_union(two_tags, tmp_path):
    db, ids = two_tags
    rows = db.files_matching_tags(["glaze", "outdoor"], mode="any")
    assert {r["id"] for r in rows} == set(ids)


def test_all_mode_is_the_intersection(two_tags, tmp_path):
    db, ids = two_tags
    rows = db.files_matching_tags(["glaze", "outdoor"], mode="all")
    assert [r["id"] for r in rows] == [ids[1]]


def test_all_mode_with_one_tag_matches_that_tag(two_tags, tmp_path):
    db, ids = two_tags
    rows = db.files_matching_tags(["glaze"], mode="all")
    assert {r["id"] for r in rows} == {ids[0], ids[1]}


def test_duplicate_tag_names_do_not_break_all_mode(two_tags):
    """count(DISTINCT) vs len(names) would disagree without the set()."""
    db, ids = two_tags
    rows = db.files_matching_tags(["glaze", "glaze"], mode="all")
    assert {r["id"] for r in rows} == {ids[0], ids[1]}


def test_unknown_tag_in_all_mode_matches_nothing(two_tags):
    db, _ids = two_tags
    assert db.files_matching_tags(["glaze", "nonexistent"], mode="all") == []


def test_empty_tag_list_returns_nothing(two_tags):
    db, _ids = two_tags
    assert db.files_matching_tags([], mode="any") == []


def test_results_carry_every_matched_tag(two_tags, tmp_path):
    db, ids = two_tags
    rows = db.files_matching_tags(["glaze", "outdoor"], mode="all")
    assert sorted(rows[0]["tags"].split(",")) == ["glaze", "outdoor"]


def test_library_view_reports_tags_per_file(two_tags):
    db, ids = two_tags
    view = {r["id"]: (r["tags"] or "") for r in db.library_view()}
    assert sorted(view[ids[1]].split(",")) == ["glaze", "outdoor"]
    assert view[ids[0]] == "glaze"
