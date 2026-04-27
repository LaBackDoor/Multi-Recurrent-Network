"""Tests for MR-KAN v2.4 - spline-shape similarity pruning."""

import torch
import torch.nn as nn

from src.model.kan import MRKAN, KANMemoryBank
from src.model.kan.pruning import cosine_similarity_fn
from src.model.kan.kan_linear import KANLinear


def test_t1_layer_link_ratios_default_matches_v1_formula():
    """Default layer_link_ratios buffer should be (i+1)/K - matches v1 inline formula."""
    bank = KANMemoryBank(
        num_items=4,
        layer_size=10,
        target_layer_sizes={1: 20},
        device=torch.device("cpu"),
    )
    expected = torch.tensor([0.25, 0.50, 0.75, 1.00], dtype=torch.float32)
    assert torch.allclose(bank.layer_link_ratios, expected), (
        f"got {bank.layer_link_ratios}, expected {expected}"
    )
    print("  [t1.1] default layer_link_ratios = (i+1)/K: PASS")


def test_t1_custom_ratios_respected():
    """Passing custom_ratios should override the default."""
    custom = torch.tensor([0.1, 0.5, 0.9], dtype=torch.float32)
    bank = KANMemoryBank(
        num_items=3,
        layer_size=10,
        target_layer_sizes={1: 20},
        custom_ratios=custom,
        device=torch.device("cpu"),
    )
    assert torch.allclose(bank.layer_link_ratios, custom)
    print("  [t1.2] custom_ratios respected: PASS")


def test_t1_custom_ratios_shape_validated():
    """custom_ratios must have shape (num_items,)."""
    bad = torch.tensor([0.5, 0.5], dtype=torch.float32)
    try:
        KANMemoryBank(
            num_items=4,
            layer_size=10,
            target_layer_sizes={1: 20},
            custom_ratios=bad,
            device=torch.device("cpu"),
        )
    except (ValueError, AssertionError):
        print("  [t1.3] custom_ratios shape validation: PASS")
        return
    raise AssertionError("expected error for shape mismatch")


def test_t1_custom_ratios_propagate_to_update_memory():
    """Constructing a bank with non-default custom_ratios should change the
    update_memory output by the corresponding factors. This is the invariant
    Task 6's pruning surgery depends on."""
    custom = torch.tensor([0.2, 0.8], dtype=torch.float32)
    bank = KANMemoryBank(
        num_items=2,
        layer_size=4,
        target_layer_sizes={1: 8},
        custom_ratios=custom,
        device=torch.device("cpu"),
    )
    memory = torch.zeros(1, 2, 4)
    new_act = torch.ones(1, 4)
    result = bank.update_memory(new_act, memory)
    # Item 0: r=0.2 -> 0.2*1 + 0.8*0 = 0.2
    # Item 1: r=0.8 -> 0.8*1 + 0.2*0 = 0.8
    assert torch.allclose(result[0, 0, :], torch.full((4,), 0.2)), result[0, 0, :]
    assert torch.allclose(result[0, 1, :], torch.full((4,), 0.8)), result[0, 1, :]
    print("  [t1.4] custom_ratios propagate to update_memory: PASS")


def test_t2_identical_kans_have_similarity_one():
    """Two KANLinears with cloned weights should produce sim ~ 1.0."""
    torch.manual_seed(0)
    kan_a = KANLinear(8, 4)
    kan_b = KANLinear(8, 4)
    # Clone weights from a to b
    kan_b.load_state_dict(kan_a.state_dict())

    refs = torch.randn(64, 8)
    sim = cosine_similarity_fn(kan_a, kan_b, refs)
    assert sim > 0.999, f"identical kans should have sim ~ 1.0, got {sim}"
    print(f"  [t2.1] identical KANs sim={sim:.6f}: PASS")


def test_t2_random_kans_have_lower_similarity():
    """Two independently random-init KANLinears should produce sim < 0.5."""
    torch.manual_seed(0)
    kan_a = KANLinear(8, 4)
    torch.manual_seed(1)
    kan_b = KANLinear(8, 4)
    refs = torch.randn(64, 8)
    sim = cosine_similarity_fn(kan_a, kan_b, refs)
    assert sim < 0.5, f"random KANs should have sim < 0.5, got {sim}"
    print(f"  [t2.2] random KANs sim={sim:.6f}: PASS")


def test_t2_validation_errors():
    """cosine_similarity_fn should raise ValueError for incompatible KANs or bad refs."""
    kan_a = KANLinear(8, 4)
    kan_b = KANLinear(8, 4)
    refs = torch.randn(32, 8)

    # in_features mismatch
    try:
        cosine_similarity_fn(KANLinear(6, 4), kan_b, torch.randn(32, 6))
        raise AssertionError("expected ValueError for in_features mismatch")
    except ValueError:
        pass

    # out_features mismatch
    try:
        cosine_similarity_fn(kan_a, KANLinear(8, 5), refs)
        raise AssertionError("expected ValueError for out_features mismatch")
    except ValueError:
        pass

    # reference_inputs wrong shape (1D)
    try:
        cosine_similarity_fn(kan_a, kan_b, torch.randn(8))
        raise AssertionError("expected ValueError for 1D reference_inputs")
    except ValueError:
        pass

    # reference_inputs wrong in_features
    try:
        cosine_similarity_fn(kan_a, kan_b, torch.randn(32, 6))
        raise AssertionError("expected ValueError for wrong reference_inputs in_features")
    except ValueError:
        pass

    print("  [t2.4] validation errors raised correctly: PASS")


def test_t2_similarity_in_unit_interval():
    """cosine_similarity_fn should always return a float in [0, 1]."""
    torch.manual_seed(2)
    for _ in range(5):
        kan_a = KANLinear(6, 3)
        kan_b = KANLinear(6, 3)
        refs = torch.randn(32, 6)
        sim = cosine_similarity_fn(kan_a, kan_b, refs)
        assert 0.0 <= sim <= 1.0, f"sim out of [0,1]: {sim}"
    print("  [t2.3] cosine_similarity_fn in [0, 1]: PASS")


def test_t3_compute_similarities_shape_and_diagonal():
    """Returned matrices have shape (K, K) and diagonal == 1.0."""
    model = MRKAN(
        nn_structure=[10, 20, 1],
        memory_structure=[4, 3, 4],
        device=torch.device("cpu"),
    )
    sims = model.compute_spline_similarities(n_samples=32)

    expected_keys = {(0, 1), (1, 1), (2, 1)}
    assert set(sims.keys()) == expected_keys, sims.keys()
    expected_K = {(0, 1): 4, (1, 1): 3, (2, 1): 4}
    for key, mat in sims.items():
        K = expected_K[key]
        assert mat.shape == (K, K), f"{key}: got {mat.shape}, expected ({K},{K})"
        assert torch.allclose(torch.diagonal(mat), torch.ones(K), atol=1e-5)
    print("  [t3.1] shape + diagonal: PASS")


def test_t3_cloned_items_detected_off_diagonal():
    """If we clone item 0's weights into item 2 within a bank, sim[0, 2] ~ 1.0."""
    torch.manual_seed(0)
    model = MRKAN(
        nn_structure=[8, 16, 1],
        memory_structure=[0, 4, 0],
        device=torch.device("cpu"),
    )
    bank = model.cell.memory_banks["1"]
    per_item = bank.memory_kans["1"]
    per_item[2].load_state_dict(per_item[0].state_dict())

    refs = {1: torch.randn(64, 16)}
    sims = model.compute_spline_similarities(reference_inputs=refs)

    mat = sims[(1, 1)]
    assert mat[0, 2] > 0.999, f"cloned items should have sim~1.0, got {mat[0, 2]}"
    assert mat[0, 1] < 0.999, f"non-cloned items should have sim<0.999, got {mat[0, 1]}"
    print(f"  [t3.2] cloned item detected: sim[0,2]={mat[0,2]:.6f}, sim[0,1]={mat[0,1]:.6f}: PASS")


def test_t3_pluggable_similarity_fn():
    """A custom similarity_fn is honored."""
    model = MRKAN(
        nn_structure=[6, 12, 1],
        memory_structure=[0, 2, 0],
        device=torch.device("cpu"),
    )

    def constant_sim(a, b, refs):
        return 0.5

    sims = model.compute_spline_similarities(
        similarity_fn=constant_sim,
        n_samples=8,
    )
    mat = sims[(1, 1)]
    assert mat[0, 1] == 0.5 and mat[1, 0] == 0.5
    assert mat[0, 0] == 1.0 and mat[1, 1] == 1.0
    print("  [t3.3] pluggable similarity_fn honored: PASS")


def main():
    print("=" * 70)
    print("MR-KAN v2.4 Tasks 1-3 tests")
    print("=" * 70)
    tests = [
        test_t1_layer_link_ratios_default_matches_v1_formula,
        test_t1_custom_ratios_respected,
        test_t1_custom_ratios_shape_validated,
        test_t1_custom_ratios_propagate_to_update_memory,
        test_t2_identical_kans_have_similarity_one,
        test_t2_random_kans_have_lower_similarity,
        test_t2_validation_errors,
        test_t2_similarity_in_unit_interval,
        test_t3_compute_similarities_shape_and_diagonal,
        test_t3_cloned_items_detected_off_diagonal,
        test_t3_pluggable_similarity_fn,
    ]
    for t in tests:
        print()
        t()
    print("\n" + "=" * 70)
    print("All MR-KAN v2.4 Tasks 1-3 tests passed.")
    print("=" * 70)


if __name__ == "__main__":
    main()
