"""Two-stage verification.

OWLv2 itself is not exercised — a 600 MB model download would make CI slow and
network-dependent. A stub verifier stands in, because the logic worth testing is
the narrowing, the ordering and the failure handling, not the detector.
"""

from pathlib import Path

import pytest

from siftr.grounding import GroundingUnavailable, Verdict, Verifier, verify_all


class _Stub:
    """Scores by filename, so expectations are readable."""

    def __init__(self, scores=None, explode=()):
        self.scores = scores or {}
        self.explode = set(explode)
        self.seen = []

    def verify(self, path, phrases):
        self.seen.append((Path(path).name, tuple(phrases)))
        if Path(path).name in self.explode:
            raise RuntimeError("decode failed")
        return Verdict(Path(path), self.scores.get(Path(path).name, 0.0), (1, 2, 3, 4))


def test_results_come_back_best_first(tmp_path):
    stub = _Stub({"a.jpg": 0.2, "b.jpg": 0.9, "c.jpg": 0.5})
    out = verify_all([tmp_path / n for n in ("a.jpg", "b.jpg", "c.jpg")], ["x"], stub)
    assert [v.path.name for v in out] == ["b.jpg", "c.jpg", "a.jpg"]


def test_every_candidate_is_checked(tmp_path):
    stub = _Stub()
    verify_all([tmp_path / f"{i}.jpg" for i in range(5)], ["a tattooed arm"], stub)
    assert len(stub.seen) == 5
    assert all(phrases == ("a tattooed arm",) for _name, phrases in stub.seen)


def test_an_unreadable_candidate_scores_zero_rather_than_failing(tmp_path):
    """One bad file must not cost the whole batch."""
    stub = _Stub({"good.jpg": 0.8}, explode={"bad.jpg"})
    out = verify_all([tmp_path / "good.jpg", tmp_path / "bad.jpg"], ["x"], stub)
    assert [v.path.name for v in out] == ["good.jpg", "bad.jpg"]
    assert out[-1].score == 0.0


def test_missing_extras_propagate_rather_than_being_swallowed(tmp_path):
    """A missing dependency is actionable; a silent zero is not."""

    class Unavailable:
        def verify(self, path, phrases):
            raise GroundingUnavailable("not installed")

    with pytest.raises(GroundingUnavailable):
        verify_all([tmp_path / "a.jpg"], ["x"], Unavailable())


def test_progress_is_reported_per_candidate(tmp_path):
    messages = []
    verify_all([tmp_path / f"{i}.jpg" for i in range(3)], ["x"], _Stub(), messages.append)
    assert len(messages) == 3
    assert "1/3" in messages[0]


def test_empty_candidate_list(tmp_path):
    assert verify_all([], ["x"], _Stub()) == []


def test_verifier_reports_a_useful_error_without_transformers(monkeypatch):
    import builtins

    real = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.startswith("transformers"):
            raise ImportError("no transformers")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(GroundingUnavailable, match="siftr\\[grounding\\]"):
        Verifier()._ensure()


# --------------------------------------------------------------- verify_tag


def test_verify_tag_narrows_then_ranks(db, tmp_path, make_images, embedder, monkeypatch):
    """CLIP picks the candidates; detection only re-ranks those."""
    from siftr.index import build_index
    from siftr.service import teach_from_paths, verify_tag

    make_images(tmp_path / "lib", (220, 40, 40), count=6, jitter=6)
    build_index(db, tmp_path / "lib", embedder, detect_faces=False)
    teach_from_paths(db, "reds", sorted((tmp_path / "lib").glob("*.png"))[:3], embedder)

    checked = []

    def fake_verify_all(paths, phrases, verifier=None, progress=None):
        checked.extend(Path(p).name for p in paths)
        return [Verdict(Path(p), 1.0 - i / 10, None) for i, p in enumerate(paths)]

    monkeypatch.setattr("siftr.grounding.verify_all", fake_verify_all)
    results = verify_tag(db, "reds", phrase="a red thing", candidates=3)

    assert len(checked) <= 3, "must not verify the whole library"
    assert results
    assert [r["verified"] for r in results] == sorted(
        (r["verified"] for r in results), reverse=True
    )
    assert all("clip" in r for r in results), "both scores kept so disagreement is visible"


def test_verify_tag_uses_the_stored_phrase(db, tmp_path, make_images, embedder, monkeypatch):
    from siftr.index import build_index
    from siftr.service import teach_from_paths, verify_tag

    make_images(tmp_path / "lib", (220, 40, 40), count=4)
    build_index(db, tmp_path / "lib", embedder, detect_faces=False)
    teach_from_paths(db, "reds", sorted((tmp_path / "lib").glob("*.png"))[:2], embedder)
    db.set_verify_phrase("reds", "a crimson square")

    used = {}

    def fake_verify_all(paths, phrases, verifier=None, progress=None):
        used["phrases"] = list(phrases)
        return [Verdict(Path(p), 0.5, None) for p in paths]

    monkeypatch.setattr("siftr.grounding.verify_all", fake_verify_all)
    verify_tag(db, "reds")
    assert used["phrases"] == ["a crimson square"]


def test_verify_tag_falls_back_to_the_tag_name(db, tmp_path, make_images, embedder, monkeypatch):
    from siftr.index import build_index
    from siftr.service import teach_from_paths, verify_tag

    make_images(tmp_path / "lib", (220, 40, 40), count=4)
    build_index(db, tmp_path / "lib", embedder, detect_faces=False)
    teach_from_paths(db, "tattoo-sleeve", sorted((tmp_path / "lib").glob("*.png"))[:2], embedder)

    used = {}

    def fake_verify_all(paths, phrases, verifier=None, progress=None):
        used["phrases"] = list(phrases)
        return [Verdict(Path(p), 0.5, None) for p in paths]

    monkeypatch.setattr("siftr.grounding.verify_all", fake_verify_all)
    verify_tag(db, "tattoo-sleeve")
    assert used["phrases"] == ["tattoo sleeve"], "hyphens are not words"


def test_verify_tag_on_an_unknown_tag_raises(db):
    from siftr.service import verify_tag

    with pytest.raises(KeyError):
        verify_tag(db, "ghost")


def test_verify_phrase_round_trips(db):
    import numpy as np

    from siftr.vectors import normalize

    db.save_concept("t", normalize(np.ones(16, dtype=np.float32)), 0.5, 2)
    assert db.set_verify_phrase("t", "a tattooed arm")
    assert db.get_concept("t")["verify_phrase"] == "a tattooed arm"
    db.set_verify_phrase("t", None)
    assert db.get_concept("t")["verify_phrase"] is None


def test_verify_considers_more_than_the_current_matches(
    db, tmp_path, make_images, embedder, monkeypatch
):
    """Verification settles uncertain cases; files already above the threshold
    are the least uncertain there are. Verifying only those would be useless."""
    from siftr.index import apply_concept, build_index
    from siftr.service import teach_from_paths, verify_tag

    # A graded library, so most files fall below a tight threshold.
    for i in range(10):
        make_images(tmp_path / "lib" / f"g{i}", (240 - i * 20, 40 + i * 12, 60), count=1)
    build_index(db, tmp_path / "lib", embedder, detect_faces=False)
    teach_from_paths(db, "reds", sorted((tmp_path / "lib").rglob("*.png"))[:2], embedder)
    matching = apply_concept(db, "reds")

    checked = []

    def fake_verify_all(paths, phrases, verifier=None, progress=None):
        checked.extend(paths)
        return [Verdict(Path(p), 0.5, None) for p in paths]

    monkeypatch.setattr("siftr.grounding.verify_all", fake_verify_all)
    verify_tag(db, "reds", phrase="red", candidates=10)

    assert len(checked) > matching, (
        f"verified {len(checked)} but the tag already matched {matching}; "
        "the borderline files are the whole point"
    )
