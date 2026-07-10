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

import pytest
import torch
import torch.nn as nn

from src.model.kan import MRKAN, KANMemoryBank
from src.model.kan.cell import sequence_kan_cache
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


def test_fused_matches_loop_gradients_with_standalone_scale_spline():
    """scale_spline=True routes the fused path through the `scaled_spline_weight`
    property (spline_weight * spline_scaler). Stacking a property's output must
    still deliver gradients to BOTH underlying parameters."""
    bank = _bank(K=3, L=6, out=4, ln=True, scale_spline=True)
    memory = torch.rand(5, 3, 6)

    def grads_via(fused):
        bank.use_fused_context = fused
        bank.zero_grad()
        mem = memory.clone().requires_grad_(True)
        bank.compute_context(mem, target_layer=1).pow(2).sum().backward()
        return {n: p.grad.clone() for n, p in bank.named_parameters()}, mem.grad.clone()

    loop_g, loop_mem_g = grads_via(fused=False)
    fused_g, fused_mem_g = grads_via(fused=True)

    scalers = [n for n in loop_g if "spline_scaler" in n]
    assert scalers, "scale_spline=True should create spline_scaler parameters"
    for name in loop_g:
        assert torch.allclose(loop_g[name], fused_g[name], atol=1e-5), (
            f"grad mismatch on {name}: "
            f"{(loop_g[name] - fused_g[name]).abs().max().item():.2e}"
        )
    assert torch.allclose(loop_mem_g, fused_mem_g, atol=1e-5)


def test_empty_bank_context_preserves_dtype():
    """A 0-item bank must not silently upcast a half-precision model to fp32."""
    bank = _bank(K=0, L=4, out=3, ln=True).half()
    memory = torch.zeros(2, 0, 4, dtype=torch.float16)
    context = bank.compute_context(memory, target_layer=1)
    assert context.dtype == torch.float16, context.dtype
    assert context.shape == (2, 3)


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


def test_stacked_params_cache_lifecycle_and_equivalence():
    """The per-sequence stack cache must not leak past the forward, must not go
    stale after a weight update, and must produce the same context as the
    uncached path."""
    torch.manual_seed(0)
    model = MRKAN([3, 8, 1], [2, 2, 2], device=torch.device("cpu"))
    banks = list(model.cell.memory_banks.values())
    x = torch.rand(2, 4, 3)

    assert all(b._kan_cache is None for b in banks)
    model(x)
    assert all(b._kan_cache is None for b in banks), "stack cache leaked past forward"

    bank = model.cell.memory_banks["0"]
    memory = torch.rand(5, bank.num_items, bank.layer_size)
    uncached = bank.compute_context(memory, target_layer=1)
    with sequence_kan_cache([bank]):
        cached = bank.compute_context(memory, target_layer=1)
        cached_again = bank.compute_context(memory, target_layer=1)
    torch.testing.assert_close(cached, uncached, rtol=0, atol=0)
    torch.testing.assert_close(cached_again, uncached, rtol=0, atol=0)

    # Same initial memory both times: any change is due to the weight update.
    state = model.init_state(2)
    before = model(x, states=state.clone()).detach().clone()
    with torch.no_grad():
        for b in banks:
            for per_item in b.memory_kans.values():
                for kan in per_item:
                    kan.base_weight.add_(0.5)
    after = model(x, states=state.clone()).detach()
    assert not torch.allclose(before, after), "stale cached parameter stacks reused"


def test_cache_released_on_exception():
    torch.manual_seed(0)
    model = MRKAN([3, 8, 1], [2, 0, 0], device=torch.device("cpu"))
    banks = list(model.cell.memory_banks.values())
    with pytest.raises(RuntimeError, match="boom"):
        with sequence_kan_cache(banks):
            assert all(b._kan_cache is not None for b in banks)
            raise RuntimeError("boom")
    assert all(b._kan_cache is None for b in banks)


def test_fused_rejects_heterogeneous_items_instead_of_diverging_silently():
    """Hand-surgery on one item's LayerNorm used to produce a silently wrong
    context (fused reads eps/LN-presence off item 0 only)."""
    bank = _bank(K=3, L=6, out=4, ln=True)
    memory = torch.rand(2, 3, 6)
    bank.use_fused_context = True

    bank.memory_kans["1"][1].layer_norm.eps = 0.1
    with pytest.raises(RuntimeError, match="shared LayerNorm eps"):
        bank.compute_context(memory, target_layer=1)
    bank.memory_kans["1"][1].layer_norm.eps = bank.memory_kans["1"][0].layer_norm.eps

    # base_activation was the one field the fused path reads off item 0 without
    # checking: a swapped activation used to diverge by ~0.5 with no error.
    bank.memory_kans["1"][2].base_activation = nn.Tanh()
    with pytest.raises(RuntimeError, match="shared base_activation"):
        bank.compute_context(memory, target_layer=1)
    bank.memory_kans["1"][2].base_activation = nn.SiLU()

    bank.memory_kans["1"][0].layer_norm = None
    with pytest.raises(RuntimeError, match="homogeneous items"):
        bank.compute_context(memory, target_layer=1)


def test_calibrate_grids_removes_hooks_when_a_batch_raises():
    """A forward hook that survives calibration keeps appending activations to
    an orphaned dict on every later forward -- an invisible memory leak."""
    torch.manual_seed(0)
    model = MRKAN([3, 8, 1], [2, 0, 0], device=torch.device("cpu"))

    def exploding_loader():
        yield torch.rand(2, 5, 3)
        raise RuntimeError("bad batch")

    def hook_count():
        return sum(
            len(m._forward_hooks)
            for m in model.modules()
            if isinstance(m, KANLinear)
        )

    assert hook_count() == 0
    with pytest.raises(RuntimeError, match="bad batch"):
        model.calibrate_grids(exploding_loader(), n_batches=4)

    assert hook_count() == 0, "calibrate_grids leaked forward hooks on exception"
    # State restore was already exception-safe; confirm it stayed that way.
    assert all(b.use_fused_context for b in model.cell.memory_banks.values())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for inductor")
@pytest.mark.parametrize("fused", [False, True])
def test_compiled_cell_matches_eager(fused):
    """torch.compile must not change the function it compiles, on either path."""
    torch._dynamo.reset()
    device = torch.device("cuda")
    torch.manual_seed(0)
    model = MRKAN([3, 8, 1], [2, 2, 2], device=device)
    model.set_fused_context(fused)
    x = torch.rand(3, 5, 3, device=device)

    def run():
        model.zero_grad(set_to_none=True)
        out = model(x, return_sequences=False)
        out.square().mean().backward()
        grads = {n: p.grad.clone() for n, p in model.named_parameters()}
        return out.detach().clone(), grads

    torch.manual_seed(1)
    out_eager, g_eager = run()

    model.cell.forward = torch.compile(model.cell.forward, dynamic=False)
    torch.manual_seed(1)
    out_compiled, g_compiled = run()

    torch.testing.assert_close(out_compiled, out_eager, rtol=1e-4, atol=1e-6)
    for name in g_eager:
        torch.testing.assert_close(
            g_compiled[name], g_eager[name], rtol=1e-3, atol=1e-5,
            msg=lambda m, n=name: f"compiled gradient mismatch on {n}\n{m}",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for inductor")
def test_compiled_model_with_cuda_graphs_matches_eager_given_a_fixed_state():
    """Whole-sequence capture under mode="reduce-overhead".

    The initial state must be passed in explicitly. Compiling MRKAN.forward
    pulls init_state()'s torch.rand into the graph, where inductor draws from
    its own RNG stream -- so a compiled forward and an eager forward with the
    same torch.manual_seed() start from *different* random memory and differ by
    ~1e-3, which looks exactly like a numerics bug and is not one.
    """
    device = torch.device("cuda")
    torch.manual_seed(0)
    x = torch.rand(4, 6, 3, device=device)

    def run(compiled):
        torch._dynamo.reset()
        torch.manual_seed(0)
        model = MRKAN([3, 8, 1], [2, 2, 2], device=device)
        state = model.init_state(4)
        if compiled:
            model.forward = torch.compile(
                model.forward, dynamic=False, mode="reduce-overhead"
            )
        outs = []
        for _ in range(3):  # replay the captured graph
            model.zero_grad(set_to_none=True)
            out = model(x, states=state.clone(), return_sequences=False)
            out.square().mean().backward()
            outs.append(out.detach().clone())
        grads = {n: p.grad.clone() for n, p in model.named_parameters()}
        return outs, grads

    eager_outs, eager_grads = run(compiled=False)
    graph_outs, graph_grads = run(compiled=True)

    for out in graph_outs[1:]:
        assert torch.equal(graph_outs[0], out), "cuda-graph replay is not stable"
    torch.testing.assert_close(graph_outs[0], eager_outs[0], rtol=1e-4, atol=1e-6)
    for name in eager_grads:
        torch.testing.assert_close(
            graph_grads[name], eager_grads[name], rtol=1e-3, atol=1e-5,
            msg=lambda m, n=name: f"cuda-graph gradient mismatch on {n}\n{m}",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for inductor")
def test_compiled_cell_sees_weight_updates_through_the_stack_cache():
    """The per-sequence stack cache holds tensors built inside the compiled
    region. If dynamo ever baked them in as constants, training would silently
    stop making progress: the loss would move but the memory KANs would not."""
    torch._dynamo.reset()
    device = torch.device("cuda")
    torch.manual_seed(0)
    model = MRKAN([3, 8, 1], [2, 2, 2], device=device)
    model.cell.forward = torch.compile(model.cell.forward, dynamic=False)
    x = torch.rand(2, 4, 3, device=device)

    torch.manual_seed(1)
    before = model(x).detach().clone()

    with torch.no_grad():
        for bank in model.cell.memory_banks.values():
            for per_item in bank.memory_kans.values():
                for kan in per_item:
                    kan.base_weight.add_(0.5)

    torch.manual_seed(1)
    after = model(x).detach()
    assert not torch.allclose(before, after), (
        "compiled forward ignored a weight update -- cached stacks were "
        "constant-folded into the graph"
    )


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
        test_calibrate_grids_removes_hooks_when_a_batch_raises,
    ]
    for t in tests:
        print()
        t()
    print("\n" + "=" * 70)
    print("All fused-context tests passed.")
    print("=" * 70)


if __name__ == "__main__":
    main()
