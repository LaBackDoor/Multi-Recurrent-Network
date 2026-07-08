"""Tests for MR-KAN.

The seven checks cover the test matrix in the MR-KAN plan (Section 5):

1. Construction sanity (canonical 3-layer)
2. Forward shape (batched, unbatched, return_sequences toggle)
3. BPTT connectivity (grad on every KAN parameter, every source layer)
4. MRN parity regression (zero-spline MR-KAN matches MRN to 1e-4)
5. State continuation (split-seq equals full-seq)
6. Chain topology correctness (5-layer)
7. LayerNorm / grid_range coupling (post-LN grid is widened by default)
"""

from typing import Dict

import torch
import torch.nn as nn

from src.model.mrn import MRN
from src.model.kan import MRKAN, MRKANCell
from src.model.kan.kan_linear import KANLinear


def _zero_all_splines(cell: MRKANCell) -> None:
    """Zero out every KANLinear's spline branch so its forward reduces to
    base_weight @ base_activation(x) (optionally after LayerNorm)."""
    with torch.no_grad():
        for bank in cell.memory_banks.values():
            for per_item in bank.memory_kans.values():
                for kan in per_item:
                    kan.spline_weight.zero_()
                    if kan.enable_standalone_scale_spline:
                        kan.spline_scaler.zero_()


def test_1_construction_sanity():
    """Canonical 3-layer construction and reported shape."""
    model = MRKAN(
        nn_structure=[10, 20, 1],
        memory_structure=[4, 3, 4],
        device=torch.device("cpu"),
    )
    assert model.input_size == 10
    assert model.output_size == 1
    assert model.num_layers == 3
    # 3 banks present (one per source layer that has a non-zero count).
    assert set(model.cell.memory_banks.keys()) == {"0", "1", "2"}

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [test 1] MRKAN [10,20,1] + [4,3,4] trainable params: {n_params:,}")
    assert n_params > 0
    print("  [test 1] construction sanity: PASS")


def test_2_forward_shape():
    """Batched, unbatched, and return_sequences toggle."""
    model = MRKAN(
        nn_structure=[8, 16, 2],
        memory_structure=[3, 2, 2],
        device=torch.device("cpu"),
    )

    # Batched
    x = torch.randn(5, 12, 8)
    y = model(x)
    assert y.shape == (5, 12, 2), y.shape

    # Unbatched
    x1 = torch.randn(12, 8)
    y1 = model(x1)
    assert y1.shape == (12, 2), y1.shape

    # return_sequences=False
    y_last = model(x, return_sequences=False)
    assert y_last.shape == (5, 2), y_last.shape

    y1_last = model(x1, return_sequences=False)
    assert y1_last.shape == (2,), y1_last.shape

    print("  [test 2] forward shapes: PASS")


def test_3_bptt_connectivity():
    """loss.backward() should populate grads on every KAN parameter at every
    source layer, including source layers distant from the hidden layer
    (e.g. output-layer memory feeding the first hidden via chain-topology)."""
    model = MRKAN(
        nn_structure=[10, 20, 1],
        memory_structure=[4, 3, 4],
        device=torch.device("cpu"),
    )

    x = torch.randn(2, 6, 10, requires_grad=False)
    y = model(x)
    loss = y.sum()
    loss.backward()

    def assert_param_has_grad(name, p):
        assert p.grad is not None, f"{name} has no grad"
        assert torch.isfinite(p.grad).all(), f"{name} grad has non-finite values"
        assert p.grad.abs().sum() > 0, f"{name} grad is all zero"

    # Plain feedforward params
    for k, p in model.cell.weights.items():
        assert_param_has_grad(f"weights[{k}]", p)
    for k, p in model.cell.biases.items():
        # Output bias might see uniform grad, check it was at least populated.
        assert p.grad is not None, f"biases[{k}] has no grad"

    # Every KANLinear across every bank, every target layer, every item
    for bank_idx, bank in model.cell.memory_banks.items():
        for tgt, per_item in bank.memory_kans.items():
            for i, kan in enumerate(per_item):
                assert_param_has_grad(
                    f"bank[{bank_idx}].kan[{tgt}][{i}].base_weight", kan.base_weight
                )
                assert_param_has_grad(
                    f"bank[{bank_idx}].kan[{tgt}][{i}].spline_weight", kan.spline_weight
                )
                if kan.layer_norm is not None:
                    assert_param_has_grad(
                        f"bank[{bank_idx}].kan[{tgt}][{i}].layer_norm.weight",
                        kan.layer_norm.weight,
                    )

    print("  [test 3] BPTT connectivity: PASS")


def test_4_mrn_parity_regression():
    """An MR-KAN with splines zeroed and base_activation=Identity should produce
    a purely-linear memory path, equivalent to MRN up to a reweighting by
    base_weight (Kaiming init). To get an apples-to-apples numerical check,
    we compare an MR-KAN (with its spline path disabled and no LayerNorm) to
    a reconstructed MRN that copies MR-KAN's base_weight as its memory weight.

    Success: the max abs diff of outputs is < 1e-4.
    """
    torch.manual_seed(0)

    nn_structure = [6, 10, 1]
    memory_structure = [2, 2, 2]
    device = torch.device("cpu")

    mrkan = MRKAN(
        nn_structure=nn_structure,
        memory_structure=memory_structure,
        kan_use_layernorm=False,        # remove LN so the base path is pure linear
        kan_base_activation=nn.Identity,  # base(x) = x, so base_output = W_base @ x
        kan_enable_standalone_scale_spline=False,
        device=device,
    )
    # Force the spline branch off so KANLinear(x) == F.linear(x, base_weight)
    _zero_all_splines(mrkan.cell)

    mrn = MRN(
        nn_structure=nn_structure,
        memory_structure=memory_structure,
        device=device,
    )

    # Copy plain feedforward weights + biases: input->hidden, hidden->output.
    with torch.no_grad():
        for k, p in mrkan.cell.weights.items():
            mrn.cell.weights[k].data.copy_(p.data)
        for k, p in mrkan.cell.biases.items():
            mrn.cell.biases[k].data.copy_(p.data)

    # Copy memory projection weights: MRN stores nn.Parameter (target_size,
    # source_size) matrices; MR-KAN stores KANLinear.base_weight with the same
    # shape. MRN's MemoryBank.compute_context does F.linear(memory, W) implicitly
    # via torch.matmul(memory, W.t()); KANLinear's forward is F.linear(x, base_weight).
    for src in mrkan.cell.memory_banks:
        mrkan_bank = mrkan.cell.memory_banks[src]
        mrn_bank = mrn.cell.memory_banks[src]
        for tgt, per_item in mrkan_bank.memory_kans.items():
            mrn_params = mrn_bank.memory_weights[tgt]
            for i, kan in enumerate(per_item):
                mrn_params[i].data.copy_(kan.base_weight.data)

    # Match initial memory by using the same random seed pattern: we init a
    # fresh state in both and run the exact same sequence.
    torch.manual_seed(42)
    x = torch.randn(3, 7, nn_structure[0])

    # Same initial state in both models. We init MR-KAN first and copy over.
    s_mrkan = mrkan.init_state(batch_size=3)
    s_mrn_banks = {k: v.clone() for k, v in s_mrkan.memory_banks.items()}
    from src.model.cell import MRNState
    s_mrn = MRNState(memory_banks=s_mrn_banks)

    y_mrkan, sf_mrkan = mrkan(x, states=s_mrkan, return_state=True)
    y_mrn, sf_mrn = mrn(x, states=s_mrn, return_state=True)

    diff = (y_mrkan - y_mrn).abs().max().item()
    print(f"  [test 4] max abs diff MRKAN(zero-spline, identity-base) vs MRN: {diff:.2e}")
    assert diff < 1e-4, f"parity failed: {diff}"
    print("  [test 4] MRN parity regression: PASS")


def test_5_state_continuation():
    """Run seq[:T] then seq[T:] with carried state == run full seq."""
    torch.manual_seed(1)
    model = MRKAN(
        nn_structure=[6, 12, 2],
        memory_structure=[2, 2, 2],
        device=torch.device("cpu"),
    )
    model.eval()

    x = torch.randn(4, 10, 6)

    # Full
    y_full, sf_full = model(x, return_state=True)

    # Split at T=4 with consistent initial state in both runs.
    s0 = model.init_state(batch_size=4)
    y_a, s_a = model(x[:, :4], states=s0, return_state=True)
    y_b, s_b = model(x[:, 4:], states=s_a, return_state=True)

    # Redo full using the same s0 for a fair comparison
    y_full_same_init, sf_full_same = model(x, states=s0, return_state=True)

    y_concat = torch.cat([y_a, y_b], dim=1)
    diff = (y_concat - y_full_same_init).abs().max().item()
    print(f"  [test 5] split vs full max abs diff: {diff:.2e}")
    assert diff < 1e-5, f"state continuation failed: {diff}"
    # Final states match too
    for k in s_b.memory_banks:
        d = (s_b.memory_banks[k] - sf_full_same.memory_banks[k]).abs().max().item()
        assert d < 1e-5, f"state[{k}] mismatch {d}"
    print("  [test 5] state continuation: PASS")


def test_6_chain_topology():
    """5-layer chain topology resolves targets per the documented rule."""
    model = MRKAN(
        nn_structure=[10, 32, 16, 8, 5],
        memory_structure=[4, 3, 2, 0, 1],
        device=torch.device("cpu"),
    )
    expected = {
        0: ["1"],  # input -> first hidden
        1: ["2"],  # hidden 1 -> hidden 2
        2: ["3"],  # hidden 2 -> hidden 3
        # layer 3 has 0 banks, so no entry
        4: ["1"],  # output -> first hidden
    }
    for src, tgts in expected.items():
        key = str(src)
        assert key in model.cell.memory_banks, f"missing bank at {src}"
        got = list(model.cell.memory_banks[key].memory_kans.keys())
        assert got == tgts, f"src={src}: expected {tgts}, got {got}"
    # No bank at layer 3
    assert "3" not in model.cell.memory_banks

    # And forward actually runs with this topology
    y = model(torch.randn(2, 5, 10))
    assert y.shape == (2, 5, 5)
    print("  [test 6] chain topology: PASS")


def test_7_layernorm_grid_coupling():
    """use_layernorm=True should widen grid_range default to (-3, 3)."""
    # Default (no LN)
    k0 = KANLinear(10, 4)
    assert k0.grid_range == (-1.0, 1.0)
    assert k0.layer_norm is None

    # LN default widens
    k1 = KANLinear(10, 4, use_layernorm=True)
    assert k1.grid_range == (-3.0, 3.0)
    assert isinstance(k1.layer_norm, nn.LayerNorm)

    # Explicit grid_range with LN is respected (user override)
    k2 = KANLinear(10, 4, use_layernorm=True, grid_range=(-1.0, 1.0))
    assert k2.grid_range == (-1.0, 1.0)

    # MRKANCell propagates the default through
    cell = MRKANCell(
        nn_structure=[8, 16, 2],
        memory_structure=[2, 2, 2],
        kan_use_layernorm=True,
        device=torch.device("cpu"),
    )
    for bank in cell.memory_banks.values():
        for per_item in bank.memory_kans.values():
            for kan in per_item:
                assert kan.grid_range == (-3.0, 3.0)
                assert kan.layer_norm is not None

    # Active grid of k1 spans exactly grid_range (not the extended support)
    active = k1.grid[:, k1.spline_order : -k1.spline_order]
    mn, mx = float(active.min()), float(active.max())
    assert abs(mn - (-3.0)) < 1e-6 and abs(mx - 3.0) < 1e-6, (mn, mx)
    print("  [test 7] LayerNorm / grid_range coupling: PASS")


def main():
    print("=" * 70)
    print("MR-KAN tests")
    print("=" * 70)
    tests = [
        test_1_construction_sanity,
        test_2_forward_shape,
        test_3_bptt_connectivity,
        test_4_mrn_parity_regression,
        test_5_state_continuation,
        test_6_chain_topology,
        test_7_layernorm_grid_coupling,
    ]
    for t in tests:
        print()
        t()
    print("\n" + "=" * 70)
    print("All MR-KAN tests passed.")
    print("=" * 70)


if __name__ == "__main__":
    main()
