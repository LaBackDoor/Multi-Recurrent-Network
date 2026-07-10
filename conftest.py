"""Pytest configuration.

Imported before any test module (and therefore before `import torch`), which is
the only point where inductor's cache location can still be set.

On Windows, inductor names fused kernels after every op it merged
("triton_poi_fused__to_copy_add_bitwise_and_div_ge_lt_mul_select_...") and nests
them under a hashed directory. Beneath the default %TEMP% -- itself a long,
8.3-expanded user path -- the total overflows MAX_PATH=260 and Triton dies with
a FileNotFoundError partway through compiling the larger graphs. Only the
torch.compile tests hit graphs big enough to trip it, so it looks like a
compile-correctness failure when it is really a path-length failure.
"""

import os
import sys

if sys.platform == "win32":
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "C:/ti")
    os.environ.setdefault("TRITON_CACHE_DIR", "C:/tc")

    # Inductor's static CUDA launcher passes 64-bit device pointers through a
    # C `long`, which is 32-bit on Windows (LLP64). Any allocation above 2 GiB
    # raises "OverflowError: Python int too large to convert to C long" from
    # torch/_inductor/runtime/static_cuda_launcher.py. It depends on where the
    # allocator happens to land, so it presents as a flaky torch.compile test
    # (observed ~1 run in 3). Fall back to the dynamic launcher.
    os.environ.setdefault("TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER", "0")
