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
