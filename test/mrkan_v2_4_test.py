"""Tests for MR-KAN v2.4 - spline-shape similarity pruning."""

import torch
import torch.nn as nn

from src.model.kan import MRKAN, KANMemoryBank


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


def main():
    print("=" * 70)
    print("MR-KAN v2.4 Task 1 tests")
    print("=" * 70)
    tests = [
        test_t1_layer_link_ratios_default_matches_v1_formula,
        test_t1_custom_ratios_respected,
        test_t1_custom_ratios_shape_validated,
        test_t1_custom_ratios_propagate_to_update_memory,
    ]
    for t in tests:
        print()
        t()
    print("\n" + "=" * 70)
    print("All MR-KAN v2.4 Task 1 tests passed.")
    print("=" * 70)


if __name__ == "__main__":
    main()
