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

`covid` config: `nn=[1,16,32,1]`, `mem=[4,3,2,4]`, B=32, T=14. Speedup vs the eager
per-item loop (`loop, eager`), same fwd / fwd+bwd column.

| variant | MR-KAN fwd | MR-KAN fwd+bwd | MRN fwd | MRN fwd+bwd |
| --- | --- | --- | --- | --- |
| loop, eager | 1.0× (103.7 ms) | 1.0× (248.4 ms) | 1.0× (11.5 ms) | 1.0× (30.9 ms) |
| fused, eager | 3.4× | 3.6× | 1.7× | 1.8× |
| fused, `compile(cell)` | 12.5× | 11.5× | 2.2× | 1.5× |
| fused, `compile(cell, reduce-overhead)` | *raises* | 23.0× | *raises* | 3.0× |
| fused, `compile(model)` | 13.4× | 11.6× | 4.2× | 4.2× |
| **fused, `compile(model, reduce-overhead)`** | **72.0×** | **80.5×** | **23.1×** | **15.1×** |

At the `ett` config (`nn=[7,64,64,1]`, `mem=[4,4,4,4]`, B=32, T=96) whole-model compile is
skipped by default; MR-KAN `fused + compile(cell, reduce-overhead)` reaches **58.9×** fwd+bwd.

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
| 25 AdamW steps, loss + final params vs eager | identical (max loss diff 6e-8, param diff 5e-5) |
| train / `no_grad` validation interleaved | works |
| ragged final batch (`drop_last=False`) | works; recompiles once (2 graphs) |
| `calibrate_grids` between epochs | works |
| per training step (fwd+bwd+AdamW) | 119.0 ms → 8.5 ms (**13.9×**), peak 23 → 17 MiB |

The two rules:

1. **Pass `states=` explicitly.** `states=None` pulls `init_state()`'s `torch.rand` into the
   graph, where inductor draws from its own RNG. Same seed, different initial memory
   (measured 3.1e-3 output difference) — which looks exactly like a numerics bug and is not one.
2. **`clone()` any output you keep across steps.** Graph replay writes into static buffers;
   retaining last step's output and reading it after the next step raises
   `accessing tensor output of CUDAGraphs that has been overwritten`. Detaching is not enough.

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
