import numpy as np

from siftr.vectors import centroid, cosine, from_blob, normalize, to_blob


def test_normalize_unit_length():
    out = normalize(np.array([3.0, 4.0]))
    assert np.isclose(np.linalg.norm(out), 1.0)


def test_normalize_leaves_zero_vector_alone():
    """A blank frame must not become NaN and poison a whole batch."""
    out = normalize(np.zeros(4))
    assert not np.isnan(out).any()
    assert np.all(out == 0)


def test_normalize_rows_of_matrix():
    out = normalize(np.array([[3.0, 4.0], [0.0, 0.0], [1.0, 0.0]]))
    assert np.allclose(np.linalg.norm(out[[0, 2]], axis=1), 1.0)
    assert np.all(out[1] == 0)


def test_blob_roundtrip_preserves_values():
    vector = normalize(np.array([0.1, -0.2, 0.3, 0.9], dtype=np.float32))
    assert np.allclose(from_blob(to_blob(vector)), vector, atol=1e-6)


def test_cosine_identical_is_one():
    v = normalize(np.array([1.0, 2.0, 3.0]))
    assert np.isclose(cosine(v, v)[0], 1.0)


def test_cosine_orthogonal_is_zero():
    assert np.isclose(cosine(np.array([1.0, 0.0]), np.array([[0.0, 1.0]]))[0], 0.0)


def test_centroid_of_identical_vectors_is_that_vector():
    v = normalize(np.array([1.0, 1.0, 0.0]))
    assert np.allclose(centroid(np.vstack([v, v, v])), v)


def test_centroid_is_normalized():
    stack = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert np.isclose(np.linalg.norm(centroid(stack)), 1.0)
