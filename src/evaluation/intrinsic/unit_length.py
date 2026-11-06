"""scale_to_unit_length: the cosine preparation the analyses share."""

import numpy as np


def scale_to_unit_length(vectors: np.ndarray) -> np.ndarray:
    """vectors scaled to unit length, so dot products are cosines and
    Euclidean distances rank pairs as cosine distance does. A zero vector
    stays zero.

    Inputs: vectors (n, d). Output: a new array of the same shape and dtype.
    Side effects: none. Exceptions: none.

    Example:
        >>> scale_to_unit_length(np.array([[3.0, 4.0]]))
        array([[0.6, 0.8]])
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    result = np.zeros_like(vectors)
    np.divide(vectors, norms, out=result, where=norms > 0)
    return result
