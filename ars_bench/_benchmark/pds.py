from __future__ import annotations
from typing import Any, Dict, List
import numpy as np

def clean_text(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()

def build_step_text(step_obj: Dict[str, Any], chain_mode: str) -> str:
    """
    chain_mode:
      - query:
          step = search_query

      - thought_query:
          step = thought + search_query
    """
    thought = clean_text(step_obj.get("thought"))
    search_query = clean_text(step_obj.get("search_query"))

    if chain_mode == "query":
        return search_query

    if chain_mode == "thought_query":
        if thought and search_query:
            return f"{thought}\nSearch query: {search_query}"
        if thought:
            return thought
        if search_query:
            return f"Search query: {search_query}"
        return ""

    raise ValueError(f"Unsupported chain_mode: {chain_mode}")

def build_reasoning_chain(example: Dict[str, Any], chain_mode: str) -> List[str]:
    steps = []

    for step_obj in example.get("step_stats", []):
        step_text = build_step_text(step_obj, chain_mode=chain_mode)
        if step_text:
            steps.append(step_text)

    return steps

def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, eps)

def transition_vectors(step_embeddings: np.ndarray) -> np.ndarray:
    """
    Given z_1, ..., z_n, return normalized transition vectors:

      d_i = normalize(z_{i+1} - z_i)

    This represents the semantic direction of each reasoning transition.
    """
    if len(step_embeddings) < 2:
        return np.zeros((0, step_embeddings.shape[-1]), dtype=np.float32)

    diff = step_embeddings[1:] - step_embeddings[:-1]
    return l2_normalize(diff).astype(np.float32)

def cosine_distance_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    a, b are assumed to be L2-normalized.
    distance = 1 - cosine_similarity
    """
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)

    sim = np.matmul(a, b.T)
    dist = 1.0 - sim
    return np.clip(dist, 0.0, 2.0).astype(np.float32)

def dtw_average_cost(a: np.ndarray, b: np.ndarray) -> float:
    """
    Dynamic Time Warping average alignment cost using cosine distance.

    This allows clean and attack chains to have different lengths.
    """
    n = len(a)
    m = len(b)

    if n == 0 and m == 0:
        return 0.0

    if n == 0 or m == 0:
        return 1.0

    cost = cosine_distance_matrix(a, b)

    dp = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
    steps = np.full((n + 1, m + 1), np.inf, dtype=np.float32)

    dp[0, 0] = 0.0
    steps[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            candidates = [
                (dp[i - 1, j], steps[i - 1, j]),
                (dp[i, j - 1], steps[i, j - 1]),
                (dp[i - 1, j - 1], steps[i - 1, j - 1]),
            ]

            prev_cost, prev_steps = min(candidates, key=lambda x: x[0])

            dp[i, j] = cost[i - 1, j - 1] + prev_cost
            steps[i, j] = prev_steps + 1.0

    return float(dp[n, m] / max(steps[n, m], 1.0))

def similarity_from_cost(cost: float) -> float:
    """
    Convert cosine-distance DTW cost to similarity.
    Cosine distance is approximately in [0, 2], so use 1 - cost / 2.
    """
    sim = 1.0 - (cost / 2.0)
    return float(np.clip(sim, 0.0, 1.0))
