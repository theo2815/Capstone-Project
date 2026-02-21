from __future__ import annotations

import numpy as np

try:
    from _eventai_cpp import batch_cosine_topk as _cpp_topk

    _HAS_CPP = True
except ImportError:
    _HAS_CPP = False


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two L2-normalized vectors."""
    return float(np.dot(a, b))


def find_matches(
    query: np.ndarray,
    database: np.ndarray,
    threshold: float = 0.4,
    top_k: int = 10,
) -> list[dict]:
    """Find top-K matches for a query embedding against a database of embeddings.

    Args:
        query: L2-normalized embedding vector, shape (512,).
        database: L2-normalized embedding matrix, shape (N, 512).
        threshold: Minimum cosine similarity to consider a match.
        top_k: Maximum number of results to return.

    Returns:
        List of dicts with 'index' and 'score' keys, sorted by score descending.
    """
    if database.shape[0] == 0:
        return []

    if _HAS_CPP:
        result = _cpp_topk(query, database, threshold, top_k)
        return [
            {"index": int(i), "score": float(s)}
            for i, s in zip(result.indices, result.scores)
        ]

    # Pure NumPy fallback
    similarities = database @ query
    mask = similarities >= threshold
    valid_indices = np.where(mask)[0]
    if len(valid_indices) == 0:
        return []

    valid_scores = similarities[valid_indices]
    top_order = np.argsort(valid_scores)[::-1][:top_k]
    return [
        {"index": int(valid_indices[i]), "score": float(valid_scores[i])}
        for i in top_order
    ]
