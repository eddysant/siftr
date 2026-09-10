"""Threshold calibration against the library.

The naive positives-only rule fails exactly where the UI needs it most: dropping
a set of similar photos onto a tag yields ~0.99 self-similarity, and a 0.99
threshold matches almost nothing. These tests pin the gap-finding behaviour that
replaces it.
"""

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
