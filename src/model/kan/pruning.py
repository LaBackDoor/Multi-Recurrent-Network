"""Pruning utilities for MR-KAN v2.4.

This module hosts:

- ``cosine_similarity_fn``: the default similarity metric (cosine on
  flattened sampled outputs) for v2.4's prune flow.
- ``PruningStats`` and ``BankPruningStats`` dataclasses (added in T4).
- ``resolve_pruning``: greedy iterative pairwise clustering (added in T4).

Similarity functions follow the contract:

    similarity_fn(kan_a, kan_b, reference_inputs) -> float in [0, 1]

where ``reference_inputs`` has shape ``(N, in_features)`` and the return is
1.0 for identical functions and 0.0 for orthogonal ones. The default uses
cosine similarity on flattened ``(N, out_features)`` outputs, clamped into
[0, 1] via ``max(0, sim)`` so the threshold semantic is monotonic.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

import torch

from src.model.kan.kan_linear import KANLinear

SimilarityFn = Callable[[KANLinear, KANLinear, torch.Tensor], float]


@torch.no_grad()
def cosine_similarity_fn(
    kan_a: KANLinear, kan_b: KANLinear, reference_inputs: torch.Tensor
) -> float:
    """Cosine similarity of two KANLinears on a shared input batch.

    Both KANs must accept ``reference_inputs`` (matching ``in_features``) and
    produce same-shape outputs.

    Returns a float in [0, 1]. Negative cosine values are clamped to 0 so
    the metric is monotonic with "more similar".
    """
    if kan_a.in_features != kan_b.in_features:
        raise ValueError(
            f"in_features mismatch: {kan_a.in_features} vs {kan_b.in_features}"
        )
    if kan_a.out_features != kan_b.out_features:
        raise ValueError(
            f"out_features mismatch: {kan_a.out_features} vs {kan_b.out_features}"
        )
    if reference_inputs.dim() != 2 or reference_inputs.size(1) != kan_a.in_features:
        raise ValueError(
            f"reference_inputs must be 2D with size [N, {kan_a.in_features}], "
            f"got {tuple(reference_inputs.shape)}"
        )

    # Flatten to a single vector per kan
    out_a = kan_a(reference_inputs).reshape(-1)
    out_b = kan_b(reference_inputs).reshape(-1)
    sim = torch.nn.functional.cosine_similarity(out_a, out_b, dim=0).item()
    # Cosine similarity is bounded in [-1, 1]; clamp negatives to 0 so the
    # metric is monotonically "more similar" (0 = orthogonal, 1 = identical).
    return max(0.0, sim)


@dataclass
class BankPruningStats:
    """Per-bank record of what was dropped and why."""

    source_layer: int
    target_layer: int
    original_K: int
    surviving_K: int
    dropped_indices: List[int]
    surviving_indices: List[int]
    similarity_matrix: torch.Tensor        # [K, K] original size
    triggering_pairs: List[Tuple[int, int, float]]  # (i, j, sim) that drove drops


@dataclass
class PruningStats:
    """Top-level pruning record returned by ``MRKAN.prune``."""

    threshold: float
    similarity_fn_name: str
    banks: Dict[Tuple[int, int], BankPruningStats]
    items_dropped: int
    items_kept: int
    params_before: int
    params_after: int


def resolve_pruning(
    sim_matrix: torch.Tensor,
    threshold: float,
) -> Tuple[List[int], List[int], List[Tuple[int, int, float]]]:
    """Iterative pairwise greedy clustering for one bank's similarity matrix.

    Walks all upper-triangle pairs sorted by similarity descending. For each
    pair (i, j) with sim > threshold and j > i, if both items are still alive
    AND dropping j would not violate the >=1-per-bank guard, marks j dropped.
    Lower-index always survives in a tie.

    Args:
        sim_matrix: square symmetric tensor [K, K] with 1.0s on diagonal.
        threshold: float in [0, 1]; pairs with sim > threshold are candidates.

    Returns:
        (dropped_indices, surviving_indices, triggering_pairs) where:
        - dropped_indices: sorted list of item indices to remove
        - surviving_indices: sorted list of item indices to keep
        - triggering_pairs: list of (i, j, sim) that drove drop decisions
    """
    K = sim_matrix.size(0)
    if K <= 1:
        return [], list(range(K)), []

    pairs = []
    for i in range(K):
        for j in range(i + 1, K):
            sim = float(sim_matrix[i, j])
            if sim > threshold:
                pairs.append((i, j, sim))

    pairs.sort(key=lambda x: x[2], reverse=True)

    alive = set(range(K))
    triggering: List[Tuple[int, int, float]] = []

    for i, j, sim in pairs:
        if i not in alive or j not in alive:
            continue
        if len(alive) <= 1:
            break
        alive.discard(j)
        triggering.append((i, j, sim))

    dropped = sorted(set(range(K)) - alive)
    surviving = sorted(alive)
    return dropped, surviving, triggering
