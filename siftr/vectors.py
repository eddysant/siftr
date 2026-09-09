"""Vector helpers.

Every embedding in siftr is stored L2-normalized, so a dot product *is* the
cosine similarity. Keeping that invariant in one place means the rest of the
codebase never has to think about magnitudes.
"""

from __future__ import annotations

import numpy as np

#: Embeddings are stored as float32 so a large library stays a reasonable size
#: on disk (a 512-d CLIP vector is 2 KB rather than the 4 KB float64 would use).
DTYPE = np.float32


def normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize a single vector or a stack of row vectors.

    Zero vectors are returned unchanged rather than producing NaNs — a blank or
    unreadable frame should not poison a whole batch.
    """
    array = np.asarray(vectors, dtype=DTYPE)
    if array.ndim == 1:
        norm = float(np.linalg.norm(array))
        return array if norm == 0.0 else (array / norm).astype(DTYPE)

    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (array / norms).astype(DTYPE)


def to_blob(vector: np.ndarray) -> bytes:
    """Serialize one embedding for SQLite storage."""
    return np.asarray(vector, dtype=DTYPE).tobytes()


def from_blob(blob: bytes) -> np.ndarray:
    """Deserialize one embedding stored by :func:`to_blob`."""
    return np.frombuffer(blob, dtype=DTYPE)


def cosine(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity between one query vector and a stack of row vectors.

    Both sides are normalized defensively; callers that already hold normalized
    vectors pay only a cheap redundant division.
    """
    query = normalize(query)
    matrix = normalize(np.atleast_2d(matrix))
    return matrix @ query


def centroid(vectors: np.ndarray) -> np.ndarray:
    """The normalized mean of a set of embeddings — a concept's "prototype".

    Averaging normalized vectors and renormalizing is the standard prototypical
    -network construction: it is robust with only a handful of examples, which
    is exactly the regime siftr operates in.
    """
    stack = normalize(np.atleast_2d(vectors))
    return normalize(stack.mean(axis=0))
