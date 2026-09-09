"""CLIP image embeddings.

Why CLIP rather than a fine-tuned classifier: siftr has to learn a concept from
a handful of examples, with no negatives and no training run. CLIP's image space
already separates visual concepts well enough that a *mean of a few examples*
(see :mod:`siftr.concepts`) is a usable detector, and the same embeddings serve
free-text search for concepts you have not taught yet.

The old implementation in this repo bolted a randomly-initialized linear layer
onto ResNet50 and never trained it, so its scores were noise. Similarity against
a real embedding space is what makes few-shot work at all.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
from PIL import Image

from .vectors import normalize

DEFAULT_MODEL = "ViT-B-32"
DEFAULT_PRETRAINED = "laion2b_s34b_b79k"


class Embedder:
    """Wraps an open_clip model and yields normalized image embeddings.

    The model is loaded lazily on first use so that ``siftr --help`` and the
    database-only commands do not pay a multi-second torch import.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        pretrained: str = DEFAULT_PRETRAINED,
        device: str | None = None,
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.pretrained = pretrained
        self.batch_size = batch_size
        self._device = device
        self._model = None
        self._preprocess = None
        self._tokenizer = None

    # ------------------------------------------------------------------ setup

    @property
    def device(self) -> str:
        if self._device is None:
            import torch

            if torch.cuda.is_available():
                self._device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                # Apple Silicon: MPS is a large speedup over CPU for ViT.
                self._device = "mps"
            else:
                self._device = "cpu"
        return self._device

    def _ensure_model(self):
        if self._model is None:
            import open_clip
            import torch

            model, _, preprocess = open_clip.create_model_and_transforms(
                self.model_name, pretrained=self.pretrained
            )
            model.eval()
            self._model = model.to(self.device)
            self._preprocess = preprocess
            self._tokenizer = open_clip.get_tokenizer(self.model_name)
            self._torch = torch
        return self._model

    @property
    def dimension(self) -> int:
        self._ensure_model()
        return int(self._model.visual.output_dim)

    # -------------------------------------------------------------- embedding

    def embed_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        """Embed PIL images, returning one normalized row vector each."""
        if not images:
            return np.empty((0, 0), dtype=np.float32)
        self._ensure_model()
        torch = self._torch

        out: list[np.ndarray] = []
        for start in range(0, len(images), self.batch_size):
            chunk = images[start : start + self.batch_size]
            tensor = torch.stack([self._preprocess(img) for img in chunk]).to(self.device)
            with torch.no_grad():
                features = self._model.encode_image(tensor)
            out.append(features.float().cpu().numpy())
        return normalize(np.vstack(out))

    def embed_paths(self, paths: Iterable[Path]) -> tuple[list[Path], np.ndarray]:
        """Embed image files, skipping any that fail to open.

        Returns the paths that succeeded alongside their embeddings so callers
        can keep the two aligned — a silently dropped image would otherwise
        shift every subsequent row.
        """
        from .media import load_image  # imported here to avoid a circular import

        loaded: list[Image.Image] = []
        kept: list[Path] = []
        for path in paths:
            try:
                loaded.append(load_image(path))
            except Exception:
                continue
            kept.append(path)
        if not loaded:
            return [], np.empty((0, 0), dtype=np.float32)
        return kept, self.embed_images(loaded)

    def embed_text(self, prompts: Sequence[str]) -> np.ndarray:
        """Embed text prompts into the same space as the images.

        This is what makes ``siftr search --text "a red bicycle"`` work with no
        examples at all; it is weaker than a taught concept but costs nothing.
        """
        self._ensure_model()
        torch = self._torch
        tokens = self._tokenizer(list(prompts)).to(self.device)
        with torch.no_grad():
            features = self._model.encode_text(tokens)
        return normalize(features.float().cpu().numpy())
