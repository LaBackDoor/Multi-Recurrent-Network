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


@pytest.mark.parametrize("nn_structure,memory_structure", CONFIGS)
def test_precomputed_input_context_matches_the_loop(nn_structure, memory_structure):
    """Hoisting the layer-0 bank's context out of the timestep loop must not
    change outputs, final memory, or gradients."""
    x = torch.randn(5, 8, nn_structure[0])

    model = _build(nn_structure, memory_structure)
    state = model.init_state(5)
    model.set_precompute_input_context(False)
    out_loop, grads_loop, mem_loop = _run(model, x, state, backward=True)

    model.set_precompute_input_context(True)
    out_pre, grads_pre, mem_pre = _run(model, x, state, backward=True)

    torch.testing.assert_close(out_pre, out_loop, rtol=1e-6, atol=1e-6)
    for key in mem_loop:
        torch.testing.assert_close(mem_pre[key], mem_loop[key], rtol=1e-6, atol=1e-6)
    for name in grads_loop:
        torch.testing.assert_close(
            grads_pre[name], grads_loop[name], rtol=1e-5, atol=1e-6,
            msg=lambda m, n=name: f"gradient mismatch on {n}\n{m}",
        )


def test_precompute_uses_the_incoming_memory_not_the_updated_one():
    """Off-by-one guard. At step t the cell reads M_t and only then writes
    M_{t+1}, so the precomputed trajectory must start at M_0 and exclude M_T.
    A trajectory shifted by one still has the right shape and would silently
    train, so pin the actual values."""
    model = _build([3, 6, 2], [2, 0, 0])
    cell = model.cell
    bank = cell.memory_banks["0"]
    x = torch.randn(4, 5, 3)
    m0 = model.init_state(4).memory_banks[0]

    context = cell.precompute_input_context(x, m0)

    # t=0 must use the untouched initial memory.
    expected_first = bank.compute_context(m0, target_layer=cell._chain_target(0))
    torch.testing.assert_close(context[0], expected_first, rtol=1e-6, atol=1e-6)

    # t=1 must use M_1 = update(inputs[:, 0], M_0), not M_0 and not M_2.
    m1 = bank.update_memory(x[:, 0], m0)
    expected_second = bank.compute_context(m1, target_layer=cell._chain_target(0))
    torch.testing.assert_close(context[1], expected_second, rtol=1e-6, atol=1e-6)
    assert not torch.allclose(context[0], context[1])


def test_precompute_respects_frozen_memory():
    """With memory updates disabled the trajectory is constant, so every
    timestep must see the initial memory."""
    model = _build([3, 6, 2], [2, 0, 0])
    model.set_update_memory(False)
    x = torch.randn(2, 6, 3)
    state = model.init_state(2)

    context = model.cell.precompute_input_context(x, state.memory_banks[0])
    for t in range(1, x.shape[1]):
        torch.testing.assert_close(context[t], context[0], rtol=0, atol=0)

    model.set_precompute_input_context(False)
    loop = model(x, states=state.clone())
    model.set_precompute_input_context(True)
    pre = model(x, states=state.clone())
    torch.testing.assert_close(pre, loop, rtol=1e-6, atol=1e-6)


def test_precompute_gradients_reach_inputs_and_initial_memory():
    """Parameter grads alone would not catch a double-counted EMA: the
    precomputed trajectory and the in-loop one are both functions of (M_0,
    inputs). Check the grads that would double if they were counted twice."""
    model = _build([4, 8, 3], [2, 2, 0])
    x0 = torch.randn(3, 6, 4)
    state0 = model.init_state(3)

    def grads(precompute):
        model.set_precompute_input_context(precompute)
        x = x0.clone().requires_grad_(True)
        m0 = state0.memory_banks[0].clone().requires_grad_(True)
        banks = {k: v.clone() for k, v in state0.memory_banks.items()}
        banks[0] = m0
        out, final = model(x, states=type(state0)(memory_banks=banks), return_state=True)
        # Consume BOTH the outputs and the returned bank-0 state, so both copies
        # of the trajectory carry gradient at once.
        (out.square().mean() + final.memory_banks[0].square().mean()).backward()
        return x.grad.clone(), m0.grad.clone()

    gx_loop, gm_loop = grads(False)
    gx_pre, gm_pre = grads(True)

    torch.testing.assert_close(gx_pre, gx_loop, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(gm_pre, gm_loop, rtol=1e-5, atol=1e-6)
    ratio = gx_pre.abs().mean() / gx_loop.abs().mean()
    assert abs(ratio - 1.0) < 1e-5, f"input grad scaled by {ratio:.4f}, not 1.0"


def test_precompute_matches_loop_under_chunked_streaming():
    """Feeding the sequence in chunks, threading the state through, must match
    both the whole-sequence result and the non-precomputed path."""
    model = _build([3, 6, 2], [2, 2, 0])
    x = torch.randn(2, 9, 3)
    state = model.init_state(2)

    def whole(precompute):
        model.set_precompute_input_context(precompute)
        return model(x, states=state.clone())

    def chunked(precompute):
        model.set_precompute_input_context(precompute)
        s = state.clone()
        outs = []
        for chunk in x.split(3, dim=1):
            out, s = model(chunk, states=s, return_state=True)
            outs.append(out)
        return torch.cat(outs, dim=1)

    reference = whole(False)
    torch.testing.assert_close(whole(True), reference, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(chunked(False), reference, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(chunked(True), reference, rtol=1e-6, atol=1e-6)


def test_precompute_falls_back_when_state_omits_the_input_bank():
    """A state without the layer-0 bank is legal; the cell's guards skip it.
    The precompute must not KeyError on it."""
    model = _build([3, 6, 2], [2, 2, 0])
    model.set_update_memory(False)
    x = torch.randn(2, 4, 3)
    full = model.init_state(2)
    partial = type(full)(memory_banks={1: full.memory_banks[1]})

    model.set_precompute_input_context(False)
    loop = model(x, states=partial)
    model.set_precompute_input_context(True)
    pre = model(x, states=partial)
    torch.testing.assert_close(pre, loop, rtol=1e-6, atol=1e-6)


def test_precompute_preserves_batch_one_state_broadcast():
    """MRN's update_memory broadcasts, so a [1, K, L] state against a batch-B
    input used to work. torch.stack would not."""
    model = _build([3, 6, 2], [2, 0, 0])
    x = torch.randn(5, 4, 3)
    state = model.init_state(1)

    model.set_precompute_input_context(False)
    loop = model(x, states=state.clone())
    model.set_precompute_input_context(True)
    pre = model(x, states=state.clone())
    assert pre.shape == (5, 4, 2)
    torch.testing.assert_close(pre, loop, rtol=1e-6, atol=1e-6)


def test_context_slabs_match_a_single_batched_call():
    """Long no_grad sequences chunk the batched context; chunking must be a
    pure implementation detail."""
    from src.model import cell as cell_mod

    model = _build([3, 6, 2], [2, 0, 0])
    bank = model.cell.memory_banks["0"]
    flat = torch.randn(37, bank.num_items, bank.layer_size)

    single = cell_mod.context_in_slabs(bank, flat, 1)
    original = cell_mod.MAX_CONTEXT_ROWS
    cell_mod.MAX_CONTEXT_ROWS = 8  # force several slabs
    try:
        slabbed = cell_mod.context_in_slabs(bank, flat, 1)
    finally:
        cell_mod.MAX_CONTEXT_ROWS = original

    assert slabbed.shape == single.shape
    # Not bit-exact: a GEMM over 8 rows tiles differently than one over 37.
    torch.testing.assert_close(slabbed, single, rtol=1e-5, atol=1e-6)


def test_precompute_is_off_by_default_for_mrn():
    """MRN's per-timestep context is already one GEMM, so the hoist does not pay
    (measured neutral-to-slower). The machinery exists and is exact; it is just
    not the default here, unlike MR-KAN."""
    model = _build([3, 6, 2], [2, 0, 0])
    assert not model.cell.can_precompute_input_context()
    model.set_precompute_input_context(True)
    assert model.cell.can_precompute_input_context()


def test_precompute_skipped_when_no_input_bank():
    model = _build([3, 6, 2], [0, 2, 0])
    model.set_precompute_input_context(True)
    assert not model.cell.can_precompute_input_context()
    x = torch.randn(2, 4, 3)
    state = model.init_state(2)
    torch.testing.assert_close(
        model(x, states=state.clone()), model(x, states=state.clone()),
        rtol=0, atol=0,
    )


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
