"""Equivalence tests for the MRN fused memory-context path.

The fused path replaces `sum_i memory[:, i, :] @ W_i.T` with a single GEMM over
the flattened items, and the per-item sluggish-update loop with one broadcast.
Both are exact algebraic rewrites, so these tests hold outputs, gradients and
memory state to tight tolerances rather than "close enough" ones.
"""

import pytest
import torch

from src.model.cell import MRNCell, sequence_weight_cache
from src.model.mrn import MRN

CPU = torch.device("cpu")

# (nn_structure, memory_structure)
CONFIGS = [
    ([4, 8, 3], [2, 2, 2]),          # canonical 3-layer
    ([1, 16, 32, 1], [4, 3, 2, 4]),  # covid-scale, 4 layers
    ([5, 6, 7, 8, 2], [3, 0, 2, 0, 1]),  # gaps in memory_structure
    ([3, 5, 2], [1, 1, 1]),          # K=1 banks
]


def _build(nn_structure, memory_structure, seed=0):
    torch.manual_seed(seed)
    return MRN(
        nn_structure=nn_structure,
        memory_structure=memory_structure,
        device=CPU,
    )


def _run(model, x, state, backward: bool):
    """Forward (and optionally backward); return output, grads, final memory."""
    model.zero_grad(set_to_none=True)
    out, final_state = model(x, states=state.clone(), return_state=True)
    grads = None
    if backward:
        out.square().mean().backward()
        grads = {
            name: p.grad.detach().clone()
            for name, p in model.named_parameters()
            if p.grad is not None
        }
    memory = {k: v.detach().clone() for k, v in final_state.memory_banks.items()}
    return out.detach().clone(), grads, memory


@pytest.mark.parametrize("nn_structure,memory_structure", CONFIGS)
def test_fused_matches_loop_forward_and_state(nn_structure, memory_structure):
    model = _build(nn_structure, memory_structure)
    x = torch.randn(6, 9, nn_structure[0])
    state = model.init_state(6)

    model.set_fused_context(False)
    out_loop, _, mem_loop = _run(model, x, state, backward=False)

    model.set_fused_context(True)
    out_fused, _, mem_fused = _run(model, x, state, backward=False)

    torch.testing.assert_close(out_fused, out_loop, rtol=1e-6, atol=1e-6)
    assert mem_loop.keys() == mem_fused.keys()
    for key in mem_loop:
        # Not bit-exact: the final memory is a function of activations, which
        # differ from the loop path by GEMM summation order. The update rule
        # itself IS bit-exact -- see test_update_memory_is_bit_exact_*.
        torch.testing.assert_close(
            mem_fused[key], mem_loop[key], rtol=1e-6, atol=1e-6
        )


@pytest.mark.parametrize("nn_structure,memory_structure", CONFIGS)
def test_fused_matches_loop_gradients(nn_structure, memory_structure):
    x = torch.randn(4, 7, nn_structure[0])

    model = _build(nn_structure, memory_structure)
    state = model.init_state(4)
    model.set_fused_context(False)
    _, grads_loop, _ = _run(model, x, state, backward=True)

    model.set_fused_context(True)
    _, grads_fused, _ = _run(model, x, state, backward=True)

    assert grads_loop.keys() == grads_fused.keys()
    assert grads_loop, "no parameter received a gradient"
    for name in grads_loop:
        torch.testing.assert_close(
            grads_fused[name], grads_loop[name], rtol=1e-5, atol=1e-6,
            msg=lambda m, n=name: f"gradient mismatch on {n}\n{m}",
        )


def test_update_memory_is_bit_exact_with_the_scalar_loop():
    """The vectorised sluggish update must reproduce the float64-derived
    ratios exactly, not merely to within an ulp."""
    bank = MRNCell([4, 5, 2], [3, 0, 0], device=CPU).memory_banks["0"]
    memory = torch.randn(3, bank.num_items, bank.layer_size)
    activation = torch.randn(3, bank.layer_size)

    expected = torch.stack(
        [
            (i + 1) / bank.num_items * activation
            + (1.0 - (i + 1) / bank.num_items) * memory[:, i, :]
            for i in range(bank.num_items)
        ],
        dim=1,
    )
    torch.testing.assert_close(
        bank.update_memory(activation, memory), expected, rtol=0, atol=0
    )


def test_weight_cache_is_released_and_not_stale():
    """The cache must not outlive the forward (it holds autograd graph), and a
    second forward after a parameter update must see the new weights."""
    model = _build([3, 6, 2], [2, 2, 2])
    x = torch.randn(2, 4, 3)
    banks = list(model.cell.memory_banks.values())

    assert all(b._weight_cache is None for b in banks)
    model(x)
    assert all(b._weight_cache is None for b in banks), "cache leaked past forward"

    # Same initial memory both times, so any output change is attributable to
    # the weight update rather than to a fresh random init_state().
    state = model.init_state(2)
    out_before = model(x, states=state.clone()).detach().clone()
    with torch.no_grad():
        for bank in banks:
            for per_item in bank.memory_weights.values():
                for w in per_item:
                    w.add_(0.5)
    out_after = model(x, states=state.clone()).detach()
    assert not torch.allclose(out_before, out_after), "stale cached weights reused"


def test_weight_cache_released_on_exception():
    model = _build([3, 6, 2], [2, 2, 2])
    banks = list(model.cell.memory_banks.values())
    with pytest.raises(RuntimeError):
        with sequence_weight_cache(banks):
            assert all(b._weight_cache is not None for b in banks)
            raise RuntimeError("boom")
    assert all(b._weight_cache is None for b in banks)


def test_cached_and_uncached_fused_agree():
    """compute_context outside a sequence_weight_cache rebuilds the concat; the
    result must match the cached path exactly."""
    model = _build([3, 6, 2], [2, 2, 2])
    bank = model.cell.memory_banks["0"]
    memory = torch.randn(5, bank.num_items, bank.layer_size)

    uncached = bank.compute_context(memory, target_layer=1)
    with sequence_weight_cache([bank]):
        cached = bank.compute_context(memory, target_layer=1)
        cached_again = bank.compute_context(memory, target_layer=1)

    torch.testing.assert_close(cached, uncached, rtol=0, atol=0)
    torch.testing.assert_close(cached_again, uncached, rtol=0, atol=0)


def test_fused_handles_unbatched_input():
    model = _build([3, 6, 2], [2, 2, 2])
    x = torch.randn(4, 3)  # [seq_len, input_size], no batch dim
    state = model.init_state(1)
    model.set_fused_context(False)
    loop = model(x, states=state.clone())
    model.set_fused_context(True)
    fused = model(x, states=state.clone())
    torch.testing.assert_close(fused, loop, rtol=1e-6, atol=1e-6)


def test_fused_under_no_grad_matches_loop():
    model = _build([3, 6, 2], [2, 2, 2])
    x = torch.randn(2, 4, 3)
    state = model.init_state(2)
    with torch.no_grad():
        model.set_fused_context(False)
        loop = model(x, states=state.clone())
        model.set_fused_context(True)
        fused = model(x, states=state.clone())
    torch.testing.assert_close(fused, loop, rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for inductor")
@pytest.mark.parametrize("unit", ["cell", "model"])
def test_compiled_matches_eager(unit):
    """torch.compile must not change the function it compiles."""
    torch._dynamo.reset()
    device = torch.device("cuda")
    torch.manual_seed(0)
    model = MRN(nn_structure=[3, 8, 2], memory_structure=[2, 2, 2], device=device)
    x = torch.randn(4, 5, 3, device=device)
    state = model.init_state(4)

    out_eager, grads_eager, mem_eager = _run(model, x, state, backward=True)

    if unit == "cell":
        model.cell.forward = torch.compile(model.cell.forward, dynamic=False)
    else:
        model.forward = torch.compile(model.forward, dynamic=False)
    out_c, grads_c, mem_c = _run(model, x, state, backward=True)

    torch.testing.assert_close(out_c, out_eager, rtol=1e-5, atol=1e-6)
    for key in mem_eager:
        torch.testing.assert_close(mem_c[key], mem_eager[key], rtol=1e-5, atol=1e-6)
    for name in grads_eager:
        torch.testing.assert_close(
            grads_c[name], grads_eager[name], rtol=1e-4, atol=1e-6,
            msg=lambda m, n=name: f"compiled gradient mismatch on {n}\n{m}",
        )
