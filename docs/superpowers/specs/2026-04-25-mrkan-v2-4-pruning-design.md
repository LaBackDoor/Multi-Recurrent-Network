# MR-KAN v2.4: Spline-Shape Similarity Pruning — Design

Status: approved
Date: 2026-04-25
Author: Abanisenioluwa Orojo (with Claude)

## Context

MR-KAN v2.1–v2.3 are implemented:
- v2.1 SL-MR-KAN: per-bank `RatioControlUnit` learns layer-link ratios
- v2.2 path-KAN toggles: `kan_input_path`, `kan_output_path`
- v2.3 `model.calibrate_grids(loader)`: safe adaptive grid update

v2.4 ports the thesis Chapter 7 pruning idea to KAN. The thesis prunes memory items whose learned layer-link ratios become similar after training, on the argument that those items have collapsed to redundant temporal regimes. In MR-KAN the layer-link ratio is replaced by a learnable univariate function (a `KANLinear`), so the natural analog of "ratios are similar" is "splines compute similar functions". v2.4 makes that operational.

## Goals

1. Identify memory items whose `KANLinear` projections compute approximately the same function.
2. Return a smaller `MRKAN` with redundant items removed and surviving weights transplanted.
3. Provide structured statistics from a single `prune()` call so ablation tables don't require redundant similarity recomputation.
4. Keep v1 and v2.1–v2.3 behavior unchanged when v2.4 features are not invoked.

## Non-goals

- Pruning whole banks (just items; ≥1 always survives per bank).
- Pruning path KANs or feedforward `nn.Linear` layers (not "items"; out of scope).
- Iterative prune-during-training. v2.4 is post-training, one-shot, matching the thesis.
- Building a generic threshold-sweep helper. Sweep loops are task-specific (eval metric, validation data, "best" criterion); we provide the primitives, the user writes the loop.
- Knowledge / rule extraction from pruned splines. Future work.

## Design decisions

### D1. Granularity: items only, with ≥1 survival guard

Pruning targets `(source_layer, target_layer, item_index)` triples. Every `(source_layer, target_layer)` pair retains at least one item. Path KANs and plain `nn.Linear` paths are not pruning targets — they are not "items" in the thesis sense.

Rationale: matches thesis Ch. 7 literally. Smaller blast radius than whole-bank pruning. Whole-bank elimination becomes the special case of pruning down to size 1, and is not pursued in v2.4.

### D2. Similarity metric: pluggable, default cosine on sampled outputs

The default similarity function is cosine similarity on flattened `[N, out_features]` outputs from N=128 reference inputs:

```
sim(kan_a, kan_b) = (vec_a · vec_b) / (||vec_a|| · ||vec_b||)
```

where `vec = kan(reference_inputs).flatten()`.

Reference inputs default to `torch.randn(N, in_features)` (matches the post-LayerNorm `N(0, 1)` distribution that `KANLinear`s see in practice). The user may pass a tensor of real activations for higher-fidelity calibration.

The metric is exposed as a `similarity_fn` parameter on the public methods so users can swap in L2 distance, Pearson correlation, or anything else for ablation. The contract is `similarity_fn(kan_a, kan_b, reference_inputs) -> float in [0, 1]`, where 1.0 means identical.

Comparison is restricted to items within the same `(source_layer, target_layer)` pair: items with different sources have different input dims, items with different targets have different output dims, so cross-bank comparison is meaningless.

### D3. Clustering: iterative pairwise (not connected-components)

Within each `(source_layer, target_layer)` bank:

1. Compute the K×K pairwise similarity matrix on the `K` items.
2. Form the list of upper-triangle pairs `(i, j)` with `i < j`.
3. Sort pairs by similarity descending.
4. Walk the sorted list. For each pair `(i, j)`:
   - If both items are still alive AND dropping `j` would not violate the ≥1-per-bank guard,
   - mark `j` dropped (lower-index survives).
   - Otherwise, skip.

Rationale: more conservative than connected-components clustering. Avoids transitive-similarity over-pruning (e.g. if `(0,1)` and `(1,2)` are both above threshold but `(0,2)` is not, connected-components would drop both 1 and 2; this algorithm drops only 1). Matches the spirit of thesis Ch. 7's pairwise SI algorithm.

### D4. Surgery: reconstruction (return new model)

`model.prune(threshold)` returns a fresh `MRKAN` with smaller `memory_structure`. The original is untouched, enabling threshold sweeps from a single training run:

```python
trained = train(MRKAN(...), data)
pruned_90, stats_90 = trained.prune(threshold=0.90)
pruned_95, stats_95 = trained.prune(threshold=0.95)
pruned_99, stats_99 = trained.prune(threshold=0.99)
# All three are independent models with real, smaller param counts.
```

Weight transplant inventory:

| Component | How it's transplanted |
|---|---|
| `weights[k]`, `biases[k]` (plain Linear paths) | Copied verbatim |
| `kan_layers[k]` (path KANs) | Copied verbatim (deep-copied state_dict) |
| `memory_kans[bank][target][i]` (memory item KANs) | Surviving items copied; ordering preserved |
| `layer_link_ratios` buffer per bank | Sliced by `surviving_indices` |
| `RCU.unit` when `learn_ratios=True` | Surgically rebuilt with new `(input_size + K_new·L)` input dim and `K_new` output dim. See D6. |

### D5. Ratio handling: preserved via `layer_link_ratios` buffer

`KANMemoryBank` is refactored to store its layer-link ratios as a buffer rather than computing `(i + 1) / K` inline.

```python
# In __init__
default_ratios = torch.tensor([(i + 1) / num_items for i in range(num_items)])
self.register_buffer("layer_link_ratios", layer_link_ratios or default_ratios)

# In update_memory (when self.rcu is None)
ratios = self.layer_link_ratios.view(1, -1, 1)  # [1, K, 1]
new_act_expanded = new_activation.unsqueeze(1).expand_as(memory)
return ratios * new_act_expanded + (1.0 - ratios) * memory
```

After pruning a bank's items `[0, 2]` from an original `K=4` bank with ratios `[0.25, 0.50, 0.75, 1.00]`, the surviving items keep ratios `[0.25, 0.75]`. Their KANLinears were trained for those temporal regimes; preserving the ratios prevents semantic drift.

This is a non-breaking refactor: when no custom ratios are supplied, `layer_link_ratios` defaults to the original `(i + 1) / K` schedule, and `update_memory` produces byte-identical output to v1. v1 parity test must continue to pass.

### D6. RCU surgery (when `learn_ratios=True`)

The RCU input is `[B, input_size + K · layer_size]` (current external input concatenated with flattened memory). Output is `[B, K]` (one logit per item). After pruning to `K_new` items, both dimensions change.

For surgery, we slice the existing weights instead of re-initializing:

**Output rows.** Keep rows of the RCU's final transform corresponding to surviving items.

**Input columns.** Keep columns `[0, input_size)` (the external-input portion) plus, for each surviving item index `i`, the columns `[input_size + i·L, input_size + (i+1)·L)` (the memory slot for that item).

This is straightforward for both backends:
- `nn.Linear`: slice `weight` along dim 0 (rows) and dim 1 (cols); slice `bias` along dim 0.
- `KANLinear`: slice `base_weight` (rows + cols), `spline_weight` (rows + cols, leave coeff dim alone), `spline_scaler` if present (rows + cols), `layer_norm.weight` and `layer_norm.bias` (the `[in_features]` axis), and `grid` buffer (`[in_features, ...]` axis).

After surgery, the surviving rows produce the same logits for the same input that they did before pruning. (The input distribution shifts because dropped items' memory slots are gone, but the function each surviving row computes is unchanged on the kept input subset.)

## Public API

Two new methods on `MRKAN`. Both delegate to the cell.

```python
def compute_spline_similarities(
    self,
    reference_inputs: Optional[Dict[int, torch.Tensor]] = None,
    similarity_fn: Optional[Callable[[KANLinear, KANLinear, torch.Tensor], float]] = None,
    n_samples: int = 128,
) -> Dict[Tuple[int, int], torch.Tensor]:
    """Pairwise within-bank similarity matrices.

    Args:
        reference_inputs: per-source-layer reference tensors of shape
            [N, layer_size]. If None, uses torch.randn(n_samples, layer_size)
            per source layer (matches post-LayerNorm distribution).
        similarity_fn: similarity callable; defaults to cosine on flattened
            outputs. Contract: returns float in [0, 1] where 1.0 means
            identical.
        n_samples: number of reference samples when reference_inputs is None.

    Returns:
        {(source_layer, target_layer): tensor[K, K]} similarity matrices.
        Diagonal is 1.0 by construction.
    """
    ...


def prune(
    self,
    threshold: float,
    reference_inputs: Optional[Dict[int, torch.Tensor]] = None,
    similarity_fn: Optional[Callable[[KANLinear, KANLinear, torch.Tensor], float]] = None,
    n_samples: int = 128,
) -> Tuple["MRKAN", "PruningStats"]:
    """Return a smaller MRKAN with redundant memory items removed.

    Original is untouched. The returned model has reduced memory_structure
    where K_new[k] = number of surviving items at source layer k.

    Args: same as compute_spline_similarities, plus threshold (float in
    [0, 1]). Items in pair (i, j) with sim > threshold and j > i are dropped
    subject to the >=1-per-bank guard; lower-index items survive.

    Returns:
        (pruned_model, stats) - see PruningStats below.
    """
    ...
```

`PruningStats` lives in a new module `src/model/kan/pruning.py`:

```python
@dataclass
class BankPruningStats:
    source_layer: int
    target_layer: int
    original_K: int
    surviving_K: int
    dropped_indices: List[int]
    surviving_indices: List[int]
    similarity_matrix: torch.Tensor      # [K, K]
    triggering_pairs: List[Tuple[int, int, float]]   # (i, j, sim) that drove drops


@dataclass
class PruningStats:
    threshold: float
    similarity_fn_name: str
    banks: Dict[Tuple[int, int], BankPruningStats]
    items_dropped: int
    items_kept: int
    params_before: int
    params_after: int
```

`PruningStats` is intentionally simple-typed (lists, ints, tensors) so it serializes cleanly to JSON for logging.

## Module layout

```
src/model/kan/
├── kan_linear.py        # unchanged
├── ratio_control.py     # unchanged
├── pruning.py           # NEW: similarity_fn, PruningStats, BankPruningStats, transplant utilities
├── cell.py              # MODIFIED: layer_link_ratios buffer, prune() method on cell
└── mrkan.py             # MODIFIED: compute_spline_similarities() and prune() on wrapper

test/
└── mrkan_v2_4_test.py   # NEW: 10 tests
```

## Test plan

`test/mrkan_v2_4_test.py`:

1. **shape**: `compute_spline_similarities` returns matrices of correct `[K, K]` shape per `(source, target)` bank; diagonal entries == 1.0.
2. **identity**: two `KANLinear` items with cloned weights produce sim ≈ 1.0; two random-init items produce sim < 1.0.
3. **trivial threshold (1.0)**: `prune(threshold=1.0)` returns a model whose `memory_structure` matches the original and stats reports 0 drops.
4. **aggressive threshold + guard**: `prune(threshold=-1.0)` (every pair passes the threshold) drops every item except the lowest-index in each bank that started with K ≥ 2. Banks that started with K = 1 are untouched. Stats reports `sum(max(0, K - 1) for K in original_memory_structure if K >= 1)` drops; resulting `memory_structure` is all 1s where the original had ≥ 1.
5. **cloned twins**: hand-construct a model where bank items 0 and 2 have identical weights (1, 3 random); verify pruning at threshold 0.99 drops exactly item 2 and not items 1 or 3, and that item 0 survives (lower-index wins).
6. **param count**: post-pruning param count exactly equals the new structure's reconstruction count (no orphans, no leaks).
7. **functional equivalence**: pruned model's forward output equals the original model's output computed with only surviving items (verified by zeroing dropped items in a clone of the original and comparing tensors with `< 1e-6` max abs diff).
8. **SL-MR-KAN compatibility**: `learn_ratios=True` model prunes correctly. Specifically: build a synthetic input whose external-input portion is fixed and whose memory portion has the dropped slots zeroed; pass it through (a) the original RCU and (b) the pruned RCU after slicing the input the same way; the surviving rows of the original output should match the pruned RCU's output to `< 1e-6` max abs diff.
9. **BPTT post-prune**: `loss.backward()` on the pruned model populates grads on every kept parameter (including KAN spline weights and RCU weights when present).
10. **v1 regression**: a v1-default model (no v2 toggles) pruned at threshold=1.0 has identical state_dict (modulo new `layer_link_ratios` buffer with default values) to the original.

## Risks

| Risk | Mitigation |
|---|---|
| Cosine similarity with random reference inputs gives high variance, leading to noisy threshold decisions | Default `n_samples=128` is large enough for stable cosine sim; doc note recommending higher N for very large output dims |
| RCU surgery silently misaligned (input column slicing wrong) | Test #8 verifies surviving rows produce identical logits, which would fail if column slicing were off |
| `layer_link_ratios` buffer refactor breaks v1 parity | Test #10 (v1 regression) and the existing v1 parity test (#4 in mrkan_test.py) are both run as part of the v2.4 acceptance |
| Path KANs accidentally pruned | Pruning loops iterate over `memory_kans` only; path KANs in `kan_layers` are deep-copied; covered by test #6 (param count check) |
| User passes `reference_inputs` for some source layers but not others | `compute_spline_similarities` falls back to random for layers not in the dict; doc explicit |

## Out-of-scope follow-ups

- **Iterative prune-during-training**. Could be useful but is a different research question (how does the network adapt to mid-training pruning?).
- **Whole-bank pruning**. If all items in a bank end up redundant, the user can `prune(threshold=0.5)` aggressively then post-process to remove banks of size 1 if desired. We don't automate.
- **Knowledge extraction**. Visualizing the surviving splines as symbolic curves could be a v3 feature.
- **Spline-shape similarity for path KANs**. Path KANs aren't items, but a future "compress the path KAN" pass could reduce its grid_size or spline_order based on shape complexity. Out of scope.
