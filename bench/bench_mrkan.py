"""Benchmark harness for MR-KAN / MRN forward and forward+backward.

Measures, on one axis, the hand-written fused memory-context path against the
per-item loop, and on another axis torch.compile (per-timestep cell graph,
whole-sequence graph, and CUDA-graph capture via mode="reduce-overhead").

Timing uses CUDA events on GPU and perf_counter on CPU. Warmup iterations are
excluded, and the first call is reported separately because torch.compile pays
compilation there.

EACH VARIANT RUNS IN ITS OWN SUBPROCESS. In-process sweeps are not trustworthy:

  * dynamo caches on the code object, so a new model instance or a grad_mode
    flip counts as a recompile. After `recompile_limit` (default 8) dynamo
    silently falls back to eager and only warns -- the "compiled" row then
    reports eager timings.
  * a mode="reduce-overhead" variant allocates a CUDA-graph private memory pool
    that outlives it (even when it raises), which measurably shifts every
    later variant's timings in the same process.

Both were observed here. Process isolation removes the whole class.

Usage:
    uv run python bench/bench_mrkan.py --device cuda --arch mrkan
    uv run python bench/bench_mrkan.py --device cuda --arch mrn
    uv run python bench/bench_mrkan.py --device cuda --explain
"""

import argparse
import contextlib
import json
import os
import subprocess
import sys
import time
from typing import Callable, Dict, List, Optional

# Must precede `import torch`: inductor reads these at import time.
#
# On Windows, inductor names fused kernels after every op it merged
# ("triton_per_fused_add_expand_mul_native_layer_norm_rsub_select_...") and
# nests them under a hashed directory. Under the default %TEMP% (itself a long
# 8.3-expanded user path) the result overflows MAX_PATH=260 and Triton dies
# with FileNotFoundError partway through compiling the larger graphs. A short
# cache root keeps the total under the limit.
if sys.platform == "win32":
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "C:/ti")
    os.environ.setdefault("TRITON_CACHE_DIR", "C:/tc")

# NOTE: inductor's static CUDA launcher is left ENABLED here even though it can
# raise "OverflowError: Python int too large to convert to C long" on Windows
# (it passes 64-bit device pointers through a 32-bit C `long`). Disabling it
# costs ~1.25x on the launch-bound compiled variants, which would understate the
# speedups this script exists to measure. `conftest.py` disables it for the test
# suite instead, where determinism matters more than throughput. If a compiled
# variant here dies with OverflowError, re-run: the failure is address-dependent
# and each variant is its own subprocess.

import torch

import src.model.kan.mrkan as mrkan_mod
import src.model.mrn as mrn_mod
from src.model.kan.mrkan import MRKAN
from src.model.mrn import MRN

# (name, nn_structure, memory_structure, batch, seq_len)
CONFIGS = [
    ("covid", [1, 16, 32, 1], [4, 3, 2, 4], 32, 14),
    ("ett", [7, 64, 64, 1], [4, 4, 4, 4], 32, 96),
]

ARCHS = {"mrkan": MRKAN, "mrn": MRN}

# Unrolling the whole sequence into one graph costs compile time superlinearly
# in T, and the backward graph roughly doubles it. Past this length the "model"
# variant is measured only when explicitly requested.
MODEL_COMPILE_MAX_T = 32

# (fused, compile_mode, hoist)
#
# Note on cudagraphs: mode="reduce-overhead" applied to the CELL raises
# "accessing tensor output of CUDAGraphs that has been overwritten". A captured
# graph writes its outputs into static buffers that the next replay clobbers,
# but the sequence loop retains every timestep's output to stack at the end.
# Graph capture therefore has to wrap the whole unrolled sequence (the "model"
# unit), where the retained tensors live inside the captured region.
VARIANTS = [
    ("loop ", "eager", True),
    ("fused", "eager", False),  # isolates the per-sequence stack hoist
    ("nopre", "eager", True),   # fused+hoist, layer-0 context NOT precomputed
    ("fused", "eager", True),
    ("loop ", "cell", True),
    ("nopre", "cell", True),
    ("fused", "cell", True),
    ("fused", "cell-cudagraph", True),  # expected to fail; kept as documentation
    ("loop ", "model", True),
    ("fused", "model", True),
    ("fused", "model-cudagraph", True),
]


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


@contextlib.contextmanager
def _no_hoist(arch: str):
    """Neutralise the per-sequence parameter-stack cache.

    Every timestep then rebuilds its own stacks, which is what the pre-hoist
    code did. Used to attribute the speedup to the hoist alone.

    Caveat for MR-KAN: this is a slight *over*statement of the hoist's win.
    Uncached, each timestep also re-runs `_validate_item_homogeneity` and
    rebuilds the stacks dict, neither of which existed in the original inline
    `_fused_context`. The MRN version is faithful (uncached `_concat_weight`
    is exactly the old inline `cat`).
    """
    mod, name = (
        (mrkan_mod, "sequence_kan_cache")
        if arch == "mrkan"
        else (mrn_mod, "sequence_weight_cache")
    )
    original = getattr(mod, name)
    setattr(mod, name, lambda banks: contextlib.nullcontext())
    try:
        yield
    finally:
        setattr(mod, name, original)


def timeit(fn: Callable[[], None], device: torch.device, warmup: int, iters: int) -> float:
    """Median-of-iters wall time in milliseconds."""
    for _ in range(warmup):
        fn()
    _sync(device)

    samples: List[float] = []
    for _ in range(iters):
        if device.type == "cuda":
            start, end = torch.cuda.Event(True), torch.cuda.Event(True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(end))
        else:
            t0 = time.perf_counter()
            fn()
            samples.append((time.perf_counter() - t0) * 1e3)
    samples.sort()
    return samples[len(samples) // 2]


def make_step(model, x: torch.Tensor, backward: bool) -> Callable[[], None]:
    if not backward:
        def step() -> None:
            with torch.no_grad():
                model(x, return_sequences=False)
        return step

    def step() -> None:
        model.zero_grad(set_to_none=True)
        out = model(x, return_sequences=False)
        out.square().mean().backward()

    return step


def explain(model, x: torch.Tensor) -> None:
    """Report dynamo graph breaks for the cell and the full unrolled model."""
    state = model.init_state(x.shape[0])
    cell_expl = torch._dynamo.explain(model.cell)(x[:, 0], state)
    print(f"    cell   : {cell_expl.graph_count} graphs, {cell_expl.graph_break_count} breaks")
    for i, reason in enumerate(cell_expl.break_reasons[:4]):
        print(f"      break[{i}]: {str(reason.reason)[:110]}")

    model_expl = torch._dynamo.explain(model)(x)
    print(f"    model  : {model_expl.graph_count} graphs, {model_expl.graph_break_count} breaks")
    for i, reason in enumerate(model_expl.break_reasons[:4]):
        print(f"      break[{i}]: {str(reason.reason)[:110]}")


def build(cfg, device: torch.device, arch: str = "mrkan", seed: int = 0):
    torch.manual_seed(seed)
    _, nn_structure, memory_structure, _, _ = cfg
    return ARCHS[arch](
        nn_structure=nn_structure,
        memory_structure=memory_structure,
        device=device,
    ).to(device)


def run_one(args) -> None:
    """Worker: measure exactly one variant and emit a JSON line on stdout."""
    torch._dynamo.config.recompile_limit = 64
    device = torch.device(args.device)
    cfg = next(c for c in CONFIGS if c[0] == args.config)
    _, nn_structure, _, batch, seq_len = cfg

    x = torch.randn(batch, seq_len, nn_structure[0], device=device)
    model = build(cfg, device, args.arch)
    # "loop"  = the true pre-optimization baseline: per-item context loop, no
    #           layer-0 hoist. Leaving precompute on here would fold part of the
    #           new speedup into the baseline and understate everything.
    # "nopre" = batched context, layer-0 still evaluated inside the loop.
    # "fused" = batched context + layer-0 precompute (the shipped default).
    model.set_fused_context(args.path in ("fused", "nopre"))
    model.set_precompute_input_context(args.path == "fused")

    if args.compile_mode == "cell":
        model.cell.forward = torch.compile(model.cell.forward, dynamic=False)
    elif args.compile_mode == "cell-cudagraph":
        model.cell.forward = torch.compile(
            model.cell.forward, dynamic=False, mode="reduce-overhead"
        )
    elif args.compile_mode == "model":
        model.forward = torch.compile(model.forward, dynamic=False)
    elif args.compile_mode == "model-cudagraph":
        model.forward = torch.compile(model.forward, dynamic=False, mode="reduce-overhead")

    ctx = _no_hoist(args.arch) if not args.hoist else contextlib.nullcontext()
    with ctx:
        step = make_step(model, x, args.backward)
        t0 = time.perf_counter()
        step()  # pays compilation
        _sync(device)
        compile_s = time.perf_counter() - t0
        ms = timeit(step, device, args.warmup, args.iters)

    # A compiled variant that produced no graphs ran as eager. Dynamo does this
    # silently (stderr warning only) once recompile_limit is exceeded, so
    # without this the row would be mislabelled as "compiled".
    graphs = torch._dynamo.utils.counters["stats"]["unique_graphs"]
    if args.compile_mode != "eager" and graphs == 0:
        raise RuntimeError(
            "compiled variant produced 0 dynamo graphs -- it silently ran eager"
        )

    print("RESULT " + json.dumps({"ms": ms, "compile_s": compile_s, "graphs": graphs}))


def _spawn(args, cfg_name: str, backward: bool, path: str, mode: str, hoist: bool):
    """Run one variant in a fresh interpreter. Returns (ms, compile_s) or None."""
    cmd = [
        sys.executable, os.path.abspath(__file__),
        "--worker",
        "--device", args.device, "--arch", args.arch, "--config", cfg_name,
        "--path", path, "--compile-mode", mode,
        "--warmup", str(args.warmup), "--iters", str(args.iters),
    ]
    if backward:
        cmd.append("--backward")
    if hoist:
        cmd.append("--hoist")

    env = dict(os.environ, PYTHONPATH=os.getcwd())
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT "):
            payload = json.loads(line[len("RESULT "):])
            return payload["ms"], payload["compile_s"]

    tail = (proc.stderr.strip().splitlines() or ["(no stderr)"])[-1]
    return None, tail[:80]


def run(args) -> None:
    device = torch.device(args.device)
    print(f"arch={args.arch}  device={device}  torch={torch.__version__}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}")
    print("(each variant runs in its own subprocess)")

    for cfg in CONFIGS:
        name, nn_structure, memory_structure, batch, seq_len = cfg
        if args.only and name != args.only:
            continue
        print(f"\n=== {name}: nn={nn_structure} mem={memory_structure} B={batch} T={seq_len} ===")

        if args.explain:
            x = torch.randn(batch, seq_len, nn_structure[0], device=device)
            explain(build(cfg, device, args.arch), x)

        skip_model = seq_len > MODEL_COMPILE_MAX_T and not args.force_model_compile
        if skip_model:
            print(f"  (skipping whole-model compile: T={seq_len} > {MODEL_COMPILE_MAX_T}; "
                  f"pass --force-model-compile to measure it)")

        results: Dict[tuple, tuple] = {}
        for backward in (False, True):
            tag = "fwd+bwd" if backward else "fwd"
            for path, mode, hoist in VARIANTS:
                if mode.startswith("model") and skip_model:
                    continue
                hoist_tag = "" if hoist else " no-hoist"
                label = f"{tag:8s} {path} {mode:15s}{hoist_tag}"

                ms, extra_info = _spawn(args, name, backward, path.strip(), mode, hoist)
                if ms is None:
                    print(f"  {label}: FAILED {extra_info}")
                    continue

                results[(tag, path, mode, hoist)] = (label, ms)
                extra = f"  (first call {extra_info:5.1f}s)" if mode != "eager" else ""
                print(f"  {label}: {ms:8.3f} ms{extra}")

        print("  -- speedup vs eager loop (same fwd / fwd+bwd) --")
        for (tag, _, _, _), (label, ms) in results.items():
            baseline = results.get((tag, "loop ", "eager", True))
            if baseline and ms:
                print(f"  {label}: {baseline[1] / ms:5.2f}x")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--arch", default="mrkan", choices=sorted(ARCHS))
    # Small models (MRN at the covid config is ~5k params) are dominated by GPU
    # clock/boost state. At warmup=3/iters=10 the same variant varied ~2x
    # run-to-run and the fused path could look slower than the unfused one.
    # These defaults bring it inside ~1%.
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--explain", action="store_true")
    p.add_argument("--force-model-compile", action="store_true")
    p.add_argument("--only", help="run just this config (e.g. covid)")
    # worker-only
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--config", default=CONFIGS[0][0])
    p.add_argument("--path", default="loop", choices=["loop", "fused", "nopre"])
    p.add_argument("--compile-mode", dest="compile_mode", default="eager")
    p.add_argument("--backward", action="store_true")
    p.add_argument("--hoist", action="store_true")
    args = p.parse_args()

    if args.worker:
        run_one(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
