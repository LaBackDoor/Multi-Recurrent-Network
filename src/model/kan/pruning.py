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

from typing import Callable

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
    return max(0.0, min(1.0, sim))
