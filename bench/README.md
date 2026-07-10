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

plus a `no-hoist` control that neutralises the per-sequence parameter-stack cache
(`sequence_kan_cache` / `sequence_weight_cache`) so its contribution can be attributed
separately.

## Results (RTX 4060, torch 2.9.1+cu130)

`covid` config: `nn=[1,16,32,1]`, `mem=[4,3,2,4]`, B=32, T=14. Speedup vs the eager
per-item loop, same fwd / fwd+bwd column.

| variant | MR-KAN fwd | MR-KAN fwd+bwd | MRN fwd | MRN fwd+bwd |
| --- | --- | --- | --- | --- |
| loop, eager | 1.0× (103.9 ms) | 1.0× (207.8 ms) | 1.0× (13.8 ms) | 1.0× (36.2 ms) |
| fused, eager, no hoist | 2.4× | 2.4× | 1.7× | 1.9× |
| fused, eager | 2.9× | 2.8× | 2.0× | 2.2× |
| loop, `compile(cell)` | 5.7× | 3.9× | 1.8× | 1.8× |
| fused, `compile(cell)` | 13.5× | 6.8× | 3.3× | 2.5× |
| fused, `compile(cell, reduce-overhead)` | *raises* | 23.7× | *raises* | 5.1× |
| fused, `compile(model)` | 10.7× | 9.1× | 4.8× | 4.6× |
| **fused, `compile(model, reduce-overhead)`** | **61.1×** | **61.8×** | **26.0×** | **24.0×** |

At the `ett` config (`nn=[7,64,64,1]`, `mem=[4,4,4,4]`, B=32, T=96) whole-model compile is
skipped by default; `fused + compile(cell, reduce-overhead)` reaches **62.7×** fwd+bwd for
MR-KAN.

Small models are dominated by GPU clock/boost state. At `warmup=3 / iters=10` the same MRN
variant varied ~2× run-to-run, and the fused path could measure *slower* than the unfused
one. The defaults (`warmup=10 / iters=30`) bring it inside ~1%. Don't lower them.

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
