"""Tests for the fused (vectorized) memory-context path.

KANMemoryBank.compute_context historically called one KANLinear per memory
item inside Python loops - O(T x banks x K) small kernel launches per
sequence. The fused path stacks the per-item parameters and computes all K
items with batched ops (one LayerNorm, one batched B-spline recursion, two
bmms). The per-item modules remain the source of truth for parameters, so
state_dicts, pruning, shrink, and interpretability are unchanged; the fused
path must therefore match the loop path numerically (fp32 kernel-order
differences only) and route gradients to the same parameters.
"""

import torch
import torch.nn as nn

from src.model.kan import MRKAN, KANMemoryBank
from src.model.kan.kan_linear import KANLinear, batched_b_splines


def _bank(K, L, out, ln=True, scale_spline=False, seed=0):
    torch.manual_seed(seed)
    return KANMemoryBank(
        num_items=K,
        layer_size=L,
        target_layer_sizes={1: out},
        kan_use_layernorm=ln,
        kan_enable_standalone_scale_spline=scale_spline,
        device=torch.device("cpu"),
    )


def _context_both_paths(bank, memory):
    bank.use_fused_context = False
    loop_out = bank.compute_context(memory, target_layer=1)
    bank.use_fused_context = True
    fused_out = bank.compute_context(memory, target_layer=1)
    return loop_out, fused_out


def test_batched_bsplines_match_single_item():
    """batched_b_splines on stacked inputs == KANLinear.b_splines per item."""
    torch.manual_seed(0)
    kans = [KANLinear(6, 4, use_layernorm=False) for _ in range(3)]
    x = torch.rand(3, 10, 6) * 2 - 1
    grids = torch.stack([k.grid for k in kans])
    batched = batched_b_splines(x, grids, kans[0].spline_order)
    for i, k in enumerate(kans):
        single = k.b_splines(x[i])
        assert torch.allclose(batched[i], single, atol=1e-7), (
            f"item {i}: batched B-spline bases diverge from single-item path"
        )
    print("  [fused] batched_b_splines matches per-item b_splines: PASS")


def test_fused_matches_loop_across_configs():
    """Outputs agree within fp32 kernel tolerance for every bank config."""
    configs = [
        dict(K=4, L=8, out=12, ln=True, scale_spline=False),
        dict(K=4, L=8, out=12, ln=False, scale_spline=False),
        dict(K=3, L=8, out=5, ln=True, scale_spline=True),
        dict(K=2, L=1, out=6, ln=True, scale_spline=False),  # LN auto-disabled
        dict(K=1, L=8, out=12, ln=True, scale_spline=False),  # single-item path
    ]
    for i, cfg in enumerate(configs):
        bank = _bank(cfg["K"], cfg["L"], cfg["out"], cfg["ln"], cfg["scale_spline"], seed=i)
        memory = torch.rand(7, cfg["K"], cfg["L"])
        loop_out, fused_out = _context_both_paths(bank, memory)
        diff = (loop_out - fused_out).abs().max().item()
        assert diff < 1e-5, f"config {cfg}: fused vs loop diff {diff:.2e}"
    print("  [fused] loop/fused agreement across configs: PASS")


def test_both_paths_arrive_at_correct_answer():
    """Loop AND fused paths against ground truths computed without
    compute_context.

    (1) Analytic: with splines zeroed, Identity base and no LayerNorm, every
        item KAN collapses to x @ W_base^T, so the bank context has a closed
        form: sum_i memory[:, i, :] @ W_i^T (plain matmuls).
    (2) Definitional: for a full nonlinear config, the context is BY
        DEFINITION sum_i KAN_i(memory[:, i, :]) - computed here by calling
        the per-item modules directly, outside the bank."""
    # --- (1) analytic linear ground truth ---
    torch.manual_seed(0)
    K, L, out, B = 3, 6, 5, 7
    bank = KANMemoryBank(
        num_items=K,
        layer_size=L,
        target_layer_sizes={1: out},
        kan_use_layernorm=False,
        kan_base_activation=nn.Identity,
        device=torch.device("cpu"),
    )
    with torch.no_grad():
        for kan in bank.memory_kans["1"]:
            kan.spline_weight.zero_()
    memory = torch.rand(B, K, L)
    expected = sum(
        torch.matmul(memory[:, i, :], bank.memory_kans["1"][i].base_weight.t())
        for i in range(K)
    )
    loop_out, fused_out = _context_both_paths(bank, memory)
    for name, got in (("loop", loop_out), ("fused", fused_out)):
        diff = (got - expected).abs().max().item()
        assert diff < 1e-6, f"{name} path wrong vs analytic answer: {diff:.2e}"

    # --- (2) definitional ground truth, full nonlinear config ---
    bank = _bank(K=4, L=8, out=6, ln=True, scale_spline=True, seed=3)
    memory = torch.rand(5, 4, 8)
    with torch.no_grad():
        expected = sum(
            bank.memory_kans["1"][i](memory[:, i, :]) for i in range(4)
        )
    loop_out, fused_out = _context_both_paths(bank, memory)
    for name, got in (("loop", loop_out), ("fused", fused_out)):
        diff = (got - expected).abs().max().item()
        assert diff < 1e-5, f"{name} path wrong vs definitional answer: {diff:.2e}"
    print("  [fused] loop & fused both match analytic + definitional answers: PASS")


def test_fused_matches_loop_gradients():
    """Gradients on every per-item parameter (and the input memory) must be
    the same regardless of which path computed the context."""
    bank = _bank(K=3, L=6, out=4, ln=True)
    memory = torch.rand(5, 3, 6)

    def grads_via(fused):
        bank.use_fused_context = fused
        bank.zero_grad()
        mem = memory.clone().requires_grad_(True)
        out = bank.compute_context(mem, target_layer=1)
        out.pow(2).sum().backward()
        param_grads = {n: p.grad.clone() for n, p in bank.named_parameters()}
        return param_grads, mem.grad.clone()

    loop_g, loop_mem_g = grads_via(fused=False)
    fused_g, fused_mem_g = grads_via(fused=True)

    assert set(loop_g) == set(fused_g)
    for name in loop_g:
        assert torch.allclose(loop_g[name], fused_g[name], atol=1e-5), (
            f"grad mismatch on {name}: "
            f"{(loop_g[name] - fused_g[name]).abs().max().item():.2e}"
        )
    assert torch.allclose(loop_mem_g, fused_mem_g, atol=1e-5)
    print("  [fused] gradient parity on all parameters + input: PASS")


def test_full_model_equivalence_and_toggle():
    """MRKAN outputs and grads agree between fused (default) and loop modes;
    set_fused_context flips every bank."""
    torch.manual_seed(0)
    model = MRKAN([4, 10, 1], [3, 2, 2], device=torch.device("cpu"))
    for bank in model.cell.memory_banks.values():
        assert bank.use_fused_context is True, "fused path should be the default"

    torch.manual_seed(1)
    x = torch.rand(2, 9, 4)
    state = model.init_state(2)

    def run(fused):
        model.set_fused_context(fused)
        model.zero_grad()
        out = model(x, states=state.clone())
        out.sum().backward()
        return out.detach().clone(), {
            n: p.grad.clone() for n, p in model.named_parameters()
        }

    out_fused, g_fused = run(True)
    out_loop, g_loop = run(False)

    diff = (out_fused - out_loop).abs().max().item()
    assert diff < 1e-5, f"model outputs diverge between paths: {diff:.2e}"
    for name in g_loop:
        assert torch.allclose(g_loop[name], g_fused[name], atol=1e-4), (
            f"model grad mismatch on {name}: "
            f"{(g_loop[name] - g_fused[name]).abs().max().item():.2e}"
        )
    # toggle actually reached the banks
    model.set_fused_context(False)
    assert all(not b.use_fused_context for b in model.cell.memory_banks.values())
    model.set_fused_context(True)
    assert all(b.use_fused_context for b in model.cell.memory_banks.values())
    print(f"  [fused] full-model output/grad parity (max out diff {diff:.2e}): PASS")


def test_fused_with_rcu_prune_and_shrink():
    """SL-MR-KAN with fused context: BPTT works, prune still operates on the
    per-item modules, and shrunk banks keep the fused default."""
    torch.manual_seed(0)
    model = MRKAN([4, 10, 1], [3, 0, 0], learn_ratios=True, device=torch.device("cpu"))
    out = model(torch.rand(2, 6, 4))
    out.sum().backward()
    for n, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), f"no/inf grad on {n}"

    bank = model.cell.memory_banks["0"]
    per_item = bank.memory_kans["1"]
    per_item[2].load_state_dict(per_item[0].state_dict())
    refs = {0: torch.rand(64, 4)}
    pruned, stats = model.prune(threshold=0.99, reference_inputs=refs)
    assert stats.banks[(0, 1)].dropped_indices == [2]
    assert pruned.cell.memory_banks["0"].use_fused_context is True
    out2 = pruned(torch.rand(2, 6, 4))
    assert out2.shape == (2, 6, 1)
    print("  [fused] SL-MR-KAN BPTT + prune + shrink under fused path: PASS")


def test_calibrate_grids_reaches_memory_kans_under_fused():
    """calibrate_grids collects inputs via forward hooks on each KANLinear;
    the fused path bypasses module calls, so calibration must drop to the
    loop path internally (and restore the caller's flags after). Guard: every
    memory KAN's grid must actually change."""
    torch.manual_seed(0)
    model = MRKAN([4, 8, 1], [2, 2, 2], device=torch.device("cpu"))
    assert all(b.use_fused_context for b in model.cell.memory_banks.values())

    before = {}
    for src, bank in model.cell.memory_banks.items():
        for tgt, per_item in bank.memory_kans.items():
            for i, kan in enumerate(per_item):
                before[(src, tgt, i)] = kan.grid.clone()

    model.calibrate_grids([torch.rand(3, 12, 4) for _ in range(2)], n_batches=2)

    unchanged = []
    for src, bank in model.cell.memory_banks.items():
        assert bank.use_fused_context is True, "fused flag not restored"
        for tgt, per_item in bank.memory_kans.items():
            for i, kan in enumerate(per_item):
                if torch.allclose(kan.grid, before[(src, tgt, i)]):
                    unchanged.append((src, tgt, i))
    assert not unchanged, (
        f"memory KAN grids never calibrated under fused default: {unchanged}"
    )
    print("  [fused] calibrate_grids still reaches every memory KAN: PASS")


def main():
    print("=" * 70)
    print("MR-KAN fused memory-context tests")
    print("=" * 70)
    tests = [
        test_batched_bsplines_match_single_item,
        test_fused_matches_loop_across_configs,
        test_both_paths_arrive_at_correct_answer,
        test_fused_matches_loop_gradients,
        test_full_model_equivalence_and_toggle,
        test_fused_with_rcu_prune_and_shrink,
        test_calibrate_grids_reaches_memory_kans_under_fused,
    ]
    for t in tests:
        print()
        t()
    print("\n" + "=" * 70)
    print("All fused-context tests passed.")
    print("=" * 70)


if __name__ == "__main__":
    main()
