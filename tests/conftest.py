"""Shared fixtures.

The tests never load CLIP or InsightFace: model downloads would make CI slow and
network-dependent, and the logic worth testing (storage, thresholds, reductions,
placement) is independent of which embedder produced the vectors. A deterministic
fake embedder stands in.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from siftr.db import Database
from siftr.vectors import normalize

DIM = 16


class FakeEmbedder:
    """Maps an image to a vector determined by its dominant colour.

    Images of a similar colour therefore embed close together, which is enough to
    exercise concept learning and search end to end.
    """

    def __init__(self, dim: int = DIM):
        self.dim = dim

    def _vector_for(self, image: Image.Image) -> np.ndarray:
        small = image.convert("RGB").resize((8, 8))
        pixels = np.asarray(small, dtype=np.float32).reshape(-1, 3).mean(axis=0)
        vector = np.zeros(self.dim, dtype=np.float32)
        # Spread the RGB signal over the vector so cosine similarity is driven by
        # hue rather than by a single coordinate.
        for i in range(self.dim):
            vector[i] = pixels[i % 3] * (1.0 + 0.01 * i)
        return normalize(vector)

    def preprocess(self, image):
        """Mirrors the real embedder's split.

        The real one returns a model-ready tensor; the shape does not matter to
        anything downstream, only that preprocess() output is what embed_tensors()
        consumes. Returning the finished vector keeps the double honest about the
        contract without pulling in torch.
        """
        return self._vector_for(image)

    def embed_tensors(self, tensors):
        if len(tensors) == 0:
            return np.empty((0, 0), dtype=np.float32)
        return normalize(np.vstack(list(tensors)))

    def embed_images(self, images):
        if not images:
            return np.empty((0, 0), dtype=np.float32)
        return self.embed_tensors([self.preprocess(img) for img in images])

    def embed_paths(self, paths):
        kept, vectors = [], []
        for path in paths:
            try:
                image = Image.open(path)
            except Exception:
                continue
            kept.append(path)
            vectors.append(self._vector_for(image))
        if not vectors:
            return [], np.empty((0, 0), dtype=np.float32)
        return kept, normalize(np.vstack(vectors))

    def embed_text(self, prompts):
        rng = np.random.default_rng(abs(hash(prompts[0])) % (2**32))
        return normalize(rng.standard_normal((len(prompts), self.dim)).astype(np.float32))


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def db(tmp_path) -> Database:
    with Database(tmp_path / "index.db") as database:
        yield database


def write_image(path, color, size=(64, 64)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


@pytest.fixture
def make_images():
    """Create a folder of solid-colour images and return the folder."""

    def _make(folder, color, count=6, jitter=12):
        for i in range(count):
            shifted = tuple(min(255, max(0, c + (i - count // 2) * jitter)) for c in color)
            write_image(folder / f"img{i:02d}.png", shifted)
        return folder

    return _make
