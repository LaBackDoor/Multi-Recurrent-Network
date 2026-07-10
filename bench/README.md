# Benchmarks

```bash
uv run python bench/bench_mrkan.py --device cuda --arch mrkan
uv run python bench/bench_mrkan.py --device cuda --arch mrn
uv run python bench/bench_mrkan.py --device cuda --explain   # dynamo graph-break census
```

## What is measured

Two independent axes:

| axis | values |
| --- | --- |
| memory-context path | `loop` (one KANLinear / matmul per memory item) vs `fused` (batched over items) |
| `torch.compile` unit | `eager`, `cell` (one timestep per graph), `model` (whole unrolled sequence), and `*-cudagraph` (`mode="reduce-overhead"`) |

plus two controls that attribute the individual optimizations: `no-hoist` neutralises the
per-sequence parameter-stack cache (`sequence_kan_cache` / `sequence_weight_cache`), and
`nopre` disables `precompute_input_context`.

The `fused` rows always enable the layer-0 precompute, including for MRN where it is
default-off, so the two architectures' rows measure the same thing. `loop` is the true
pre-optimization baseline: per-item context loop, no hoist, no precompute.

## Results (RTX 4060, torch 2.9.1+cu130)

`covid` config: `nn=[1,16,32,1]`, `mem=[4,3,2,4]`, B=32, T=14. Median ms, one sweep.

| variant | MR-KAN fwd | MR-KAN fwd+bwd | MRN fwd | MRN fwd+bwd |
| --- | --- | --- | --- | --- |
| loop, eager (baseline) | 104.1 | 211.3 | 15.3 | 31.3 |
| fused, eager | 30.6 | 67.3 | 6.6 | 18.0 |
| fused, `compile(cell)` | 8.0 | 26.8 | 5.0 | 15.5 |
| fused, `compile(cell, reduce-overhead)` | *raises* | 13.8 | *raises* | 8.8 |
| fused, `compile(model)` | 7.6 | 20.5 | 2.7 | 11.0 |
| **fused, `compile(model, reduce-overhead)`** | **2.3** | **4.1** | **0.52** | **2.1** |

Roughly: hand-fusion ~3× (MR-KAN) / ~1.8× (MRN); `compile(cell)` on top of that another ~2.5×;
whole-sequence capture another ~5×. End to end **~50× (MR-KAN) and ~15× (MRN) on fwd+bwd**.

**Quote the absolute compiled times, not the multipliers.** The compiled rows reproduce to ~1%,
but the *eager baselines* drift across sessions: MR-KAN's `fwd+bwd loop eager` measured 211,
248 and 209 ms on three different days, and one MRN repeat jumped 30.2 → 44.3 ms. Any speedup
computed against them inherits that ±20–40%. Treat the ratios as indicative.

At the `ett` config (`nn=[7,64,64,1]`, `mem=[4,4,4,4]`, B=32, T=96) whole-model compile is
skipped by default; MR-KAN `fused + compile(cell, reduce-overhead)` runs fwd+bwd in 68 ms
against a ~4.0 s eager loop.

The `nopre` variant isolates `precompute_input_context` (the layer-0 hoist). It is a real win
for MR-KAN — 1.45× (covid fwd) and a repeated-measurement 1.2× (ett, both directions) — because
each per-timestep context there is ~20 kernels of B-spline work. For MRN it is neutral to ~5%
slower, since MRN's context is already a single GEMM; it is therefore **default-on for MR-KAN
and default-off for MRN** (`set_precompute_input_context(True)` to enable).

Small models are dominated by GPU clock/boost state. At `warmup=3 / iters=10` the same MRN
variant varied ~2× run-to-run, and the fused path could measure *slower* than the unfused
one. The defaults (`warmup=10 / iters=30`) bring it inside ~1%. Don't lower them, and don't
trust a single row that contradicts the others — re-run it. (One `ett fwd` row in a full sweep
read 385 ms where three back-to-back repeats read 207–221 ms.)

## Is `reduce-overhead` usable for training?

`bench/cudagraph_training_probe.py` drives the things a real loop does — optimizer steps,
interleaved `no_grad` validation, a ragged final batch, `calibrate_grids` between epochs —
rather than a tight `step(); step()` loop. Verdict: **yes, with two rules.**

| scenario | result |
| --- | --- |
| 25 AdamW steps, loss + final params vs eager | identical (max loss diff 3e-8, param diff 1e-5) |
| gradient accumulation (4 backwards, 1 step) | raises unless `p.grad` is cloned each backward; exact once it is |
| train / `no_grad` validation interleaved | works |
| ragged final batch (`drop_last=False`) | works; recompiles once (2 graphs) |
| `calibrate_grids` between epochs | works |
| per training step (fwd+bwd+AdamW) | ~10–18× faster than eager, peak 23 → 17 MiB |

The two rules:

1. **Pass `states=` explicitly.** `states=None` pulls `init_state()`'s `torch.rand` into the
   graph, where inductor draws from its own RNG. Same seed, different initial memory
   (measured 3.1e-3 output difference) — which looks exactly like a numerics bug and is not one.
2. **`clone()` anything you keep across steps — including `p.grad`.** Graph replay writes into
   static buffers. Retaining last step's output and reading it after the next step raises
   `accessing tensor output of CUDAGraphs that has been overwritten`; `detach()` is not enough,
   because it shares storage. Gradients are outputs of the captured *backward*, so micro-batch
   accumulation hits the same wall: without `p.grad = p.grad.clone()` after each backward it
   raises at `opt.step()`; with it, the accumulated gradients match eager to 1.2e-7.

Both failures are loud. We could not construct a case where graph capture silently produced
wrong gradients in an ordinary loop.

`T` is pinned by the capture, and a batch-size change costs one recompile. Both are fine for a
fixed-window trainer. Throughput above is a *lower bound*: the probe must disable inductor's
static CUDA launcher (see below), which costs ~1.25× on these launch-bound paths.

## Methodology notes

**Every variant runs in its own subprocess.** An in-process sweep is not trustworthy:

- Dynamo caches on the *code object*. A new model instance, or flipping `grad_mode`
  between the fwd-only and fwd+bwd measurements, counts as a recompile. Once
  `recompile_limit` (default 8) is hit, dynamo **falls back to eager and only warns** —
  so the "compiled" rows silently report eager timings.
- A `mode="reduce-overhead"` variant allocates a CUDA-graph private memory pool that
  outlives it, *even when it raises*, and measurably shifts every subsequent variant in
  the same process.

Both were observed on this repo. Process isolation removes the class of problem, and
also makes a crashing variant (see below) harmless to its neighbours.

Timing is a median of `--iters` runs, CUDA-event timed, after `--warmup` untimed runs.
The first call is reported separately since `torch.compile` pays compilation there.

## Known: `reduce-overhead` cannot wrap the cell

`torch.compile(cell, mode="reduce-overhead")` raises

```
RuntimeError: Error: accessing tensor output of CUDAGraphs that has been overwritten
```

A captured graph writes its outputs into static buffers that the next replay clobbers,
but the sequence loop retains every timestep's output (and memory state) to stack at the
end. CUDA-graph capture therefore has to wrap the **whole unrolled sequence**, where the
retained tensors live inside the captured region. The `cell-cudagraph` variant is kept in
the sweep as executable documentation of this.

## Windows

`bench_mrkan.py` sets `TORCHINDUCTOR_CACHE_DIR=C:/ti` and `TRITON_CACHE_DIR=C:/tc` before
importing torch. Inductor names fused kernels after every op it merged, and under the
default `%TEMP%` the hashed path overflows `MAX_PATH=260`, killing Triton mid-compile with
a `FileNotFoundError`. It only bites on the larger graphs, so a smoke test can pass while
the real workload fails.

Triton itself comes from the `triton-windows` wheel, declared in `pyproject.toml` behind a
`sys_platform == 'win32'` marker (on Linux, `torch` pulls in `triton` itself).

A compiled variant may intermittently die with
`OverflowError: Python int too large to convert to C long`. Inductor's static CUDA launcher
passes 64-bit device pointers through a C `long`, which is 32 bits on Windows (LLP64), so
any allocation above 2 GiB overflows — it depends on where the allocator lands, so it looks
random. `conftest.py` sets `TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER=0` for the **test suite**,
where determinism matters. The benchmark deliberately leaves it on, because disabling it
costs ~1.25x on the launch-bound compiled variants and would understate the speedups. Each
variant is its own subprocess, so just re-run if one trips.
