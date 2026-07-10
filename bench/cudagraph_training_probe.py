"""Is torch.compile(model, mode="reduce-overhead") usable in a real MR-KAN training loop?

The whole-sequence CUDA-graph variant is by far the fastest thing the benchmark
finds, but graph capture imposes constraints (static shapes, static input
addresses, no retained outputs across replays) that a benchmark's tight
`step(); step(); step()` loop never exercises. This probe drives the things a
real training loop actually does, and reports which ones survive.

    uv run python bench/cudagraph_training_probe.py

Each scenario prints PASS / FAIL / DIFFERS with the evidence.
"""

import os
import sys

if sys.platform == "win32":
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "C:/ti")
    os.environ.setdefault("TRITON_CACHE_DIR", "C:/tc")
    # Inductor's static CUDA launcher passes 64-bit device pointers through a
    # C `long`, 32 bits on Windows (LLP64). In the short benchmark loops this
    # trips intermittently; in a real training loop (optimizer state, larger
    # pool) it fires on EVERY scenario. It is not optional here.
    #
    # Consequence: the throughput number below is a LOWER BOUND. The dynamic
    # launcher costs ~1.25x on exactly these launch-bound paths.
    os.environ.setdefault("TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER", "0")

import torch

from src.model.kan.mrkan import MRKAN

DEVICE = torch.device("cuda")
NN, MEM = [4, 24, 16, 1], [4, 3, 2, 4]
B, T = 16, 12
STEPS = 25


def _data(n_batches, batch=B, seq=T, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return [
        (
            torch.rand(batch, seq, NN[0], generator=g).to(DEVICE),
            torch.rand(batch, NN[-1], generator=g).to(DEVICE),
        )
        for _ in range(n_batches)
    ]


def _reset():
    """torch._dynamo.reset() does NOT clear torch._dynamo.utils.counters, so a
    counter read after it would include the previous scenario's graphs."""
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    torch.cuda.reset_peak_memory_stats()


def _cudagraph_skips() -> int:
    return sum(torch._dynamo.utils.counters["inductor"].get(k, 0)
               for k in ("cudagraph_skips",))


def _assert_graphs_engaged(label: str) -> bool:
    """mode="reduce-overhead" degrades to plain inductor WITHOUT raising when
    capture is unsafe. Every scenario below would then 'PASS' while testing
    nothing it claims to test."""
    skips = _cudagraph_skips()
    if skips:
        print(f"    !! cudagraphs SKIPPED ({skips}) in {label} -- ran plain inductor")
        return False
    return True


def _build(compiled: bool, seed: int = 0):
    _reset()
    torch.manual_seed(seed)
    model = MRKAN(NN, MEM, device=DEVICE)
    if compiled:
        model.forward = torch.compile(model.forward, dynamic=False, mode="reduce-overhead")
    return model


def _train(model, batches, fixed_state, lr=1e-3):
    """Returns the loss at every step."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    for x, y in batches:
        opt.zero_grad(set_to_none=True)
        out = model(x, states=fixed_state.clone(), return_sequences=False)
        loss = torch.nn.functional.mse_loss(out, y)
        loss.backward()
        opt.step()
        losses.append(loss.item())  # forces a sync every step, as real code does
    return losses


def scenario_loss_parity():
    """The whole point: does a compiled+captured training run learn the same
    thing as an eager one?

    Loss trajectories alone are neither necessary nor sufficient (compensating
    errors under AdamW can keep losses close while parameters drift), so the
    verdict requires BOTH loss parity and final-parameter parity.
    """
    batches = _data(STEPS)
    eager = _build(False)
    state = eager.init_state(B)

    eager_losses = _train(eager, batches, state)
    graph = _build(True)
    graph_losses = _train(graph, batches, state)
    engaged = _assert_graphs_engaged("loss parity")

    drift = max(abs(a - b) for a, b in zip(eager_losses, graph_losses))
    pdiff = max(
        (pe - pg).abs().max().item()
        for (_, pe), (_, pg) in zip(eager.named_parameters(), graph.named_parameters())
    )
    ok = engaged and drift < 1e-4 and pdiff < 1e-4
    print(f"  loss parity over {STEPS} optimizer steps: {'PASS' if ok else 'DIFFERS'}")
    print(f"    max |loss diff| over the run  {drift:.3e}")
    print(f"    max |param diff| after training {pdiff:.3e}")
    print(f"    eager  loss {eager_losses[0]:.6f} -> {eager_losses[-1]:.6f}")
    print(f"    graph  loss {graph_losses[0]:.6f} -> {graph_losses[-1]:.6f}")
    return ok


def scenario_retained_output_is_overwritten():
    """The hazard that killed reduce-overhead on the CELL, tested at the model
    level: a captured graph writes outputs into static buffers that the next
    replay clobbers. If you hold last step's output tensor and read it after
    the next step, do you get stale/garbage data, or does torch protect you?

    Doubles as a tripwire: if cudagraphs were silently skipped, the retained
    tensor stays valid and this reports 'not captured'.
    """
    model = _build(True)
    state = model.init_state(B)
    batches = _data(3)

    outs = []
    for x, _ in batches:
        with torch.no_grad():
            outs.append(model(x, states=state.clone(), return_sequences=False))

    if not _assert_graphs_engaged("retained output"):
        return False
    try:
        stale = outs[0].clone()
        fresh = outs[-1].clone()
        differs = not torch.allclose(stale, fresh)
        print(f"  retained outputs across replays: readable "
              f"(differ={differs}) -- clone() before keeping them")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  retained outputs across replays: RAISES {type(exc).__name__}: "
              f"{str(exc)[:80]} -- you must clone() every step")
        return True  # informative either way; not a failure of the approach


def scenario_interleaved_validation():
    """Training steps interleaved with no_grad validation, as every real loop does.
    Forward-only under a captured graph is where the cell-level variant died."""
    batches = _data(6)
    val = _data(2, seed=99)
    model = _build(True)
    state = model.init_state(B)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    try:
        for i, (x, y) in enumerate(batches):
            model.train()
            opt.zero_grad(set_to_none=True)
            out = model(x, states=state.clone(), return_sequences=False)
            torch.nn.functional.mse_loss(out, y).backward()
            opt.step()
            if i % 2 == 1:
                model.eval()
                with torch.no_grad():
                    for vx, vy in val:
                        vout = model(vx, states=state.clone(), return_sequences=False)
                        _ = torch.nn.functional.mse_loss(vout, vy).item()
        engaged = _assert_graphs_engaged("train/eval interleaving")
        print(f"  train/eval interleaving: {'PASS' if engaged else 'ran without graphs'}")
        return engaged
    except Exception as exc:  # noqa: BLE001
        print(f"  train/eval interleaving: FAIL {type(exc).__name__}: {str(exc)[:110]}")
        return False


def scenario_ragged_last_batch():
    """DataLoader(drop_last=False) hands you a short final batch. Static shapes
    mean a recompile; does it merely recompile, or does it break?"""
    model = _build(True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    try:
        for batch in (B, B, B // 2):  # the short one last
            state = model.init_state(batch)
            x, y = _data(1, batch=batch)[0]
            opt.zero_grad(set_to_none=True)
            out = model(x, states=state.clone(), return_sequences=False)
            torch.nn.functional.mse_loss(out, y).backward()
            opt.step()
        # Counters were cleared in _build -> _reset, so this counts only this
        # scenario's graphs.
        graphs = torch._dynamo.utils.counters["stats"]["unique_graphs"]
        skips = _cudagraph_skips()
        print(f"  ragged final batch: PASS (recompiled; {graphs} unique graphs, "
              f"{skips} cudagraph skips)")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  ragged final batch: FAIL {type(exc).__name__}: {str(exc)[:110]}")
        return False


def scenario_implicit_state_rng():
    """states=None pulls init_state()'s torch.rand into the graph, where
    inductor draws from its own RNG. Same seed, different initial memory."""
    x = _data(1)[0][0]
    eager = _build(False)
    torch.manual_seed(7)
    out_eager = eager(x, return_sequences=False).detach()

    graph = _build(True)
    torch.manual_seed(7)
    out_graph = graph(x, return_sequences=False).detach()

    diff = (out_eager - out_graph).abs().max().item()
    print(f"  states=None, same seed: max |diff| vs eager {diff:.3e}"
          f"  ({'matches' if diff < 1e-5 else 'DIFFERS -- pass states= explicitly'})")
    return diff < 1e-5


def scenario_calibrate_grids_between_epochs():
    """calibrate_grids mutates knot grids and spline weights in place, toggles
    use_fused_context, and runs forwards through the compiled callable."""
    model = _build(True)
    state = model.init_state(B)
    batches = _data(3)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    try:
        for x, y in batches:
            opt.zero_grad(set_to_none=True)
            out = model(x, states=state.clone(), return_sequences=False)
            torch.nn.functional.mse_loss(out, y).backward()
            opt.step()

        grid_before = model.cell.memory_banks["0"].memory_kans["1"][0].grid.clone()
        model.calibrate_grids([b[0] for b in batches], n_batches=2)
        grid_after = model.cell.memory_banks["0"].memory_kans["1"][0].grid
        moved = not torch.allclose(grid_before, grid_after)

        # And training must still work, seeing the NEW grids.
        x, y = batches[0]
        opt.zero_grad(set_to_none=True)
        out = model(x, states=state.clone(), return_sequences=False)
        torch.nn.functional.mse_loss(out, y).backward()
        opt.step()
        print(f"  calibrate_grids between epochs: PASS (grid moved={moved})")
        return moved
    except Exception as exc:  # noqa: BLE001
        print(f"  calibrate_grids between epochs: FAIL {type(exc).__name__}: {str(exc)[:110]}")
        return False


def scenario_throughput():
    batches = _data(STEPS)
    results, peaks = {}, {}
    for name, compiled in (("eager", False), ("model-cudagraph", True)):
        model = _build(compiled)  # resets peak-memory stats
        state = model.init_state(B)
        _train(model, batches[:5], state)  # warmup / compile
        torch.cuda.synchronize()
        events = torch.cuda.Event(True), torch.cuda.Event(True)
        events[0].record()
        _train(model, batches, state)
        events[1].record()
        torch.cuda.synchronize()
        results[name] = events[0].elapsed_time(events[1]) / len(batches)
        peaks[name] = torch.cuda.max_memory_allocated() / 2**20

    speedup = results["eager"] / results["model-cudagraph"]
    print(f"  per training step (fwd+bwd+AdamW): eager {results['eager']:.3f} ms  "
          f"cudagraph {results['model-cudagraph']:.3f} ms  ({speedup:.2f}x)")
    print(f"  peak allocated: eager {peaks['eager']:.0f} MiB  "
          f"cudagraph {peaks['model-cudagraph']:.0f} MiB")
    return speedup > 1.0


def main() -> None:
    print(f"torch {torch.__version__}  gpu {torch.cuda.get_device_name(0)}")
    print(f"model nn={NN} mem={MEM} B={B} T={T}\n")
    scenarios = [
        ("loss parity", scenario_loss_parity),
        ("retained outputs", scenario_retained_output_is_overwritten),
        ("interleaved validation", scenario_interleaved_validation),
        ("ragged last batch", scenario_ragged_last_batch),
        ("implicit state RNG", scenario_implicit_state_rng),
        ("calibrate_grids", scenario_calibrate_grids_between_epochs),
        ("throughput", scenario_throughput),
    ]
    verdict = {}
    for title, fn in scenarios:
        print(f"[{title}]")
        try:
            verdict[title] = fn()
        except Exception as exc:  # noqa: BLE001
            print(f"  UNCAUGHT {type(exc).__name__}: {str(exc)[:140]}")
            verdict[title] = False
        print()
    print("summary:", {k: ("ok" if v else "no") for k, v in verdict.items()})


if __name__ == "__main__":
    main()
