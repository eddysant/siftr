"""Learning a visual concept from a folder of examples.

The model is deliberately simple: a concept is the normalized mean of its
examples' embeddings (a "prototype"), and a file matches when its cosine
similarity to that prototype clears a threshold. With five to fifty examples and
no negatives, a prototype beats anything with more parameters — there is not
enough data to fit them, and this cannot overfit or diverge.

Choosing the threshold is the part that actually needs care, and it is why
supplying negatives is worth it when you can.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .embed import Embedder
from .media import IMAGE_EXTENSIONS
from .vectors import centroid, cosine

#: Used when a concept is taught from positives alone. CLIP cosine similarities
#: between unrelated images sit near 0.5-0.6 and same-concept pairs run higher,
#: so a fixed floor keeps a loose positives-only concept from matching the whole
#: library.
MIN_THRESHOLD = 0.60


@dataclass
class Concept:
    name: str
    prototype: np.ndarray
    threshold: float
    n_examples: int
    #: Mean similarity of the examples to their own prototype. A low value means
    #: the example folder is visually incoherent and the concept will be vague.
    cohesion: float = 0.0


def example_images(folder: Path) -> list[Path]:
    """Every image directly usable as an example, searched recursively."""
    folder = Path(folder).expanduser()
    if not folder.exists():
        raise FileNotFoundError(f"example folder does not exist: {folder}")
    found = sorted(
        p
        for p in folder.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
        and not any(part.startswith(".") for part in p.parts)
    )
    if not found:
        raise ValueError(f"no images found in {folder}")
    return found


def learn(
    name: str,
    positives: Path,
    embedder: Embedder,
    negatives: Path | None = None,
    percentile: float = 10.0,
    margin: float = 0.01,
) -> Concept:
    """Learn a concept from an example folder.

    Threshold selection:

    * With **negatives**, the threshold is placed between the two score
      distributions — just above the strongest negative, but never above the
      ``percentile``-th weakest positive. This is the accurate path: it is fit to
      what actually distinguishes your concept from your library.
    * With **positives only**, the threshold is the ``percentile``-th percentile
      of the positives' own similarity to their prototype, floored at
      :data:`MIN_THRESHOLD`. That accepts most of the examples' own spread while
      refusing to go so loose that everything matches.

    ``percentile=10`` means "expect to catch about 90% of things as similar to
    the prototype as the examples are", trading a little recall for precision.
    """
    paths = example_images(positives)
    kept, vectors = embedder.embed_paths(paths)
    if len(kept) == 0:
        raise ValueError(f"none of the {len(paths)} images in {positives} could be read")

    prototype = centroid(vectors)
    positive_scores = cosine(prototype, vectors)
    cohesion = float(positive_scores.mean())

    floor = float(np.percentile(positive_scores, percentile))

    if negatives is not None:
        _, negative_vectors = embedder.embed_paths(example_images(negatives))
        if len(negative_vectors):
            negative_scores = cosine(prototype, negative_vectors)
            # Clear the hardest negative, but never demand more of a match than
            # the examples themselves manage — otherwise one odd negative that
            # genuinely resembles the concept would push the threshold above the
            # positives and the concept would match nothing.
            threshold = min(float(negative_scores.max()) + margin, floor)
            return Concept(name, prototype, threshold, len(kept), cohesion)

    return Concept(name, prototype, max(floor, MIN_THRESHOLD), len(kept), cohesion)


def score_against(prototype: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return cosine(prototype, matrix)


def best_per_file(file_ids: Sequence[int], scores: np.ndarray) -> dict[int, float]:
    """Collapse per-frame scores to one score per file, keeping the maximum.

    A concept appearing in a single sampled frame of a video is a real match for
    that file, so max is the right reduction — averaging would bury it.
    """
    best: dict[int, float] = {}
    for file_id, score in zip(file_ids, scores, strict=True):
        value = float(score)
        if value > best.get(file_id, float("-inf")):
            best[file_id] = value
    return best
