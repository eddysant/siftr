import numpy as np
import pytest

from siftr.concepts import MIN_THRESHOLD, best_per_file, example_images, learn


def test_example_images_finds_nested_images(tmp_path, make_images):
    make_images(tmp_path / "a", (200, 40, 40), count=3)
    make_images(tmp_path / "b" / "c", (200, 40, 40), count=2)
    assert len(example_images(tmp_path)) == 5


def test_example_images_ignores_hidden_and_non_images(tmp_path, make_images):
    make_images(tmp_path, (10, 10, 200), count=2)
    (tmp_path / "notes.txt").write_text("hello")
    (tmp_path / ".hidden.png").write_bytes(b"x")
    assert len(example_images(tmp_path)) == 2


def test_example_images_rejects_missing_folder(tmp_path):
    with pytest.raises(FileNotFoundError):
        example_images(tmp_path / "nope")


def test_example_images_rejects_empty_folder(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError):
        example_images(tmp_path / "empty")


def test_learn_reports_example_count_and_cohesion(tmp_path, make_images, embedder):
    make_images(tmp_path / "red", (220, 30, 30), count=6, jitter=4)
    concept = learn("red", tmp_path / "red", embedder)
    assert concept.n_examples == 6
    # Near-identical examples should be highly cohesive.
    assert concept.cohesion > 0.99


def test_positives_only_threshold_respects_the_floor(tmp_path, make_images, embedder):
    """Tight examples would yield a ~1.0 threshold; the floor is a lower bound only."""
    make_images(tmp_path / "red", (220, 30, 30), count=6, jitter=2)
    concept = learn("red", tmp_path / "red", embedder)
    assert concept.threshold >= MIN_THRESHOLD


def test_incoherent_examples_still_get_the_floor(tmp_path, embedder, make_images):
    from PIL import Image

    folder = tmp_path / "mixed"
    folder.mkdir()
    for i, color in enumerate([(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]):
        Image.new("RGB", (32, 32), color).save(folder / f"{i}.png")
    concept = learn("mixed", folder, embedder)
    assert concept.threshold >= MIN_THRESHOLD
    assert concept.cohesion < 0.99


def test_negatives_pull_the_threshold_above_them(tmp_path, make_images, embedder):
    make_images(tmp_path / "red", (230, 20, 20), count=6, jitter=3)
    make_images(tmp_path / "blue", (20, 20, 230), count=6, jitter=3)

    with_negatives = learn("red", tmp_path / "red", embedder, negatives=tmp_path / "blue")
    from siftr.vectors import cosine

    _, negative_vectors = embedder.embed_paths(example_images(tmp_path / "blue"))
    worst_negative = float(cosine(with_negatives.prototype, negative_vectors).max())
    assert with_negatives.threshold > worst_negative


def test_negatives_never_push_threshold_past_the_positives(tmp_path, embedder):
    """A negative that genuinely resembles the concept must not zero out recall."""
    from PIL import Image

    positives = tmp_path / "pos"
    negatives = tmp_path / "neg"
    positives.mkdir()
    negatives.mkdir()
    for i in range(5):
        Image.new("RGB", (32, 32), (200, 40, 40)).save(positives / f"{i}.png")
    # An identical image as a "negative" — the pathological case.
    Image.new("RGB", (32, 32), (200, 40, 40)).save(negatives / "same.png")

    concept = learn("red", positives, embedder, negatives=negatives)
    _, positive_vectors = embedder.embed_paths(example_images(positives))
    from siftr.vectors import cosine

    scores = cosine(concept.prototype, positive_vectors)
    assert (scores >= concept.threshold).any(), "threshold must still admit some positives"


def test_learn_rejects_a_folder_of_unreadable_images(tmp_path, embedder):
    folder = tmp_path / "broken"
    folder.mkdir()
    (folder / "a.png").write_bytes(b"not actually a png")
    with pytest.raises(ValueError):
        learn("broken", folder, embedder)


def test_best_per_file_keeps_the_maximum():
    """A concept in one frame of a video makes the whole file a match."""
    best = best_per_file([1, 1, 1, 2], np.array([0.10, 0.95, 0.20, 0.30]))
    assert best == pytest.approx({1: 0.95, 2: 0.30})


def test_best_per_file_handles_empty():
    assert best_per_file([], np.array([])) == {}
