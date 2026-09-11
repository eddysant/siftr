"""Threshold calibration against the library.

The naive positives-only rule fails exactly where the UI needs it most: dropping
a set of similar photos onto a tag yields ~0.99 self-similarity, and a 0.99
threshold matches almost nothing. These tests pin the gap-finding behaviour that
replaces it.
"""

from pathlib import Path

import numpy as np
import pytest

from siftr.concepts import MIN_THRESHOLD, calibrate_threshold


def test_finds_the_gap_between_matches_and_the_rest():
    """The realistic shape: a tight cluster of matches far above the bulk."""
    positives = np.array([0.99, 0.99, 0.98, 0.99, 0.99])
    library = np.array([0.99, 0.98, 0.98, 0.97, 0.55, 0.54, 0.52, 0.50, 0.48, 0.45])

    threshold = calibrate_threshold(positives, library)

    assert 0.55 < threshold < 0.97, f"threshold {threshold} should sit in the gap"
    assert (library >= threshold).sum() == 4, "should admit exactly the cluster"


def test_tight_examples_do_not_produce_an_unusable_threshold():
    """The bug this function exists to fix."""
    positives = np.full(6, 0.995)
    library = np.array([0.99, 0.99, 0.98, 0.98, 0.98, 0.60, 0.58, 0.55, 0.52, 0.50])

    threshold = calibrate_threshold(positives, library)

    assert threshold < 0.98, "must not demand near-identity"
    assert (library >= threshold).sum() == 5


def test_falls_back_when_there_is_no_library():
    positives = np.array([0.9, 0.85, 0.8])
    threshold = calibrate_threshold(positives, np.array([]))
    assert threshold == pytest.approx(max(np.percentile(positives, 10), MIN_THRESHOLD))


def test_falls_back_on_a_tiny_library():
    """Too few samples to locate a boundary honestly."""
    positives = np.array([0.9, 0.9, 0.9])
    assert calibrate_threshold(positives, np.array([0.5, 0.6])) >= MIN_THRESHOLD


def test_falls_back_when_scores_grade_smoothly():
    """No real boundary exists, so do not invent one."""
    positives = np.full(5, 0.95)
    library = np.linspace(0.94, 0.61, 40)  # evenly spaced, no gap
    threshold = calibrate_threshold(positives, library)
    assert threshold == pytest.approx(max(np.percentile(positives, 10), MIN_THRESHOLD))


def test_never_returns_below_the_floor():
    positives = np.full(5, 0.99)
    library = np.concatenate([np.full(5, 0.99), np.full(20, 0.10)])
    assert calibrate_threshold(positives, library) >= MIN_THRESHOLD


def test_never_rejects_the_examples_themselves():
    """A threshold above the positives would match nothing at all."""
    positives = np.array([0.88, 0.90, 0.86, 0.91, 0.89])
    library = np.concatenate([np.array([0.90, 0.89, 0.87]), np.full(20, 0.62)])
    threshold = calibrate_threshold(positives, library)
    assert threshold <= float(np.percentile(positives, 10))


def test_works_when_the_tag_matches_most_of_the_library():
    """A percentile rule would break here; a gap rule does not."""
    positives = np.full(5, 0.97)
    library = np.concatenate([np.full(30, 0.96), np.full(6, 0.55)])
    threshold = calibrate_threshold(positives, library)
    assert (library >= threshold).sum() == 30


def test_works_when_the_tag_matches_almost_nothing():
    positives = np.full(5, 0.97)
    library = np.concatenate([np.array([0.95]), np.full(40, 0.50)])
    threshold = calibrate_threshold(positives, library)
    assert (library >= threshold).sum() == 1


# ------------------------------------------- examples must not calibrate themselves


def test_library_scores_omits_the_examples(db, tmp_path, make_images, embedder):
    """The fix itself: the files a prototype was built from must not appear in
    the pool used to calibrate its threshold."""
    from siftr.index import build_index
    from siftr.service import _library_scores
    from siftr.vectors import centroid

    make_images(tmp_path / "lib", (200, 40, 40), count=10)
    build_index(db, tmp_path / "lib", embedder, detect_faces=False)

    everything = sorted((tmp_path / "lib").glob("*.png"))
    examples = everything[:4]
    _kept, vectors = embedder.embed_paths(examples)
    prototype = centroid(vectors)

    assert len(_library_scores(db, prototype)) == 10
    assert len(_library_scores(db, prototype, examples)) == 6


def test_teaching_excludes_its_own_examples(db, tmp_path, make_images, embedder, monkeypatch):
    """End to end: teach_from_paths must pass the examples through to be excluded."""
    from siftr.index import build_index
    from siftr.service import teach_from_paths

    make_images(tmp_path / "lib", (200, 40, 40), count=10)
    build_index(db, tmp_path / "lib", embedder, detect_faces=False)
    examples = sorted((tmp_path / "lib").glob("*.png"))[:4]

    seen = {}
    import siftr.service as service_mod

    original = service_mod._library_scores

    def spy(db_, prototype, exclude=()):
        seen["exclude"] = list(exclude)
        return original(db_, prototype, exclude)

    monkeypatch.setattr(service_mod, "_library_scores", spy)
    teach_from_paths(db, "x", examples, embedder)

    assert [Path(p).name for p in seen["exclude"]] == [p.name for p in examples]


def test_the_real_world_distribution_that_broke_calibration():
    """Numbers measured on a real 27-photo library.

    Five examples scored 0.857-0.889 against their own prototype and were the top
    five in the library; the next photo down was 0.716. With the examples left in
    the pool the widest gap was that 0.14 between them and everything else, so the
    threshold came out at 0.786 and the tag matched only its own examples — 0/8 on
    held-out matches. Excluded, the same data gives a usable cutoff.
    """
    positives = np.array([0.857, 0.864, 0.865, 0.879, 0.889])
    rest = np.array(
        [
            0.716,
            0.705,
            0.680,
            0.649,
            0.632,
            0.591,
            0.558,
            0.523,
            0.486,
            0.454,
            0.445,
            0.422,
            0.420,
            0.409,
            0.404,
            0.391,
            0.387,
            0.383,
            0.346,
            0.343,
            0.343,
            0.254,
        ]
    )

    with_examples = calibrate_threshold(positives, np.concatenate([positives, rest]))
    without = calibrate_threshold(positives, rest)

    assert with_examples > rest.max(), "reproduces the bug: nothing but the examples matches"
    assert without <= rest.max(), "excluded, the threshold admits real matches"


def test_dimension_works_without_visual_output_dim():
    """SigLIP's timm-backed tower has no `visual.output_dim`; reading it directly
    made every SigLIP model unusable."""

    from siftr.embed import Embedder

    class Tower:
        embed_dim = 768

    class Model:
        visual = Tower()

    e = Embedder()
    e._model = Model()
    e._ensure_model = lambda: Model()
    assert e.dimension == 768


def test_dimension_falls_back_to_measuring():
    """A tower advertising nothing at all still has to yield a width."""
    import numpy as np

    from siftr.embed import Embedder

    class Bare:
        pass

    class Model:
        visual = Bare()

    e = Embedder()
    e._model = Model()
    e._ensure_model = lambda: Model()
    e.embed_images = lambda images: np.zeros((1, 1152), dtype=np.float32)
    assert e.dimension == 1152
