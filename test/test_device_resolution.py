"""Tests for device resolution (CUDA > MPS > CPU) and MPS operability.

default_device() decides where models land when no device is passed. The MPS
tier needs one extra guarantee: torch.linalg.lstsq has no MPS kernel, so
curve2coeff (used by update_grid / calibrate_grids) must transparently solve
on CPU for MPS-resident models.
"""

import torch

from src.model.device import default_device
from src.model.kan import MRKAN
from src.model.kan.kan_linear import KANLinear


def test_default_device_preference_order():
    """CUDA when available; else MPS; else CPU."""
    orig_cuda = torch.cuda.is_available
    orig_mps = torch.backends.mps.is_available
    try:
        torch.cuda.is_available = lambda: True
        torch.backends.mps.is_available = lambda: True
        assert default_device().type == "cuda"

        torch.cuda.is_available = lambda: False
        assert default_device().type == "mps"

        torch.backends.mps.is_available = lambda: False
        assert default_device().type == "cpu"
    finally:
        torch.cuda.is_available = orig_cuda
        torch.backends.mps.is_available = orig_mps
    print("  [device] preference order CUDA > MPS > CPU: PASS")


def test_models_construct_and_run_on_default_device():
    """No-device construction must land every parameter on default_device()
    and forward cleanly there."""
    expected = default_device()
    model = MRKAN([4, 8, 1], [2, 2, 2])
    devices = {p.device.type for p in model.parameters()}
    assert devices == {expected.type}, (
        f"expected all params on {expected.type}, got {devices}"
    )
    out = model(torch.rand(2, 5, 4, device=expected))
    assert out.shape == (2, 5, 1) and torch.isfinite(out).all()
    print(f"  [device] default-device model runs on {expected.type}: PASS")


def test_update_grid_works_on_mps():
    """curve2coeff must fall back to a CPU lstsq for MPS tensors."""
    if not torch.backends.mps.is_available():
        print("  [device] MPS not available: SKIP")
        return
    kan = KANLinear(4, 3, use_layernorm=False).to("mps")
    x = torch.rand(64, 4, device="mps") * 2 - 1
    kan.update_grid(x)
    assert kan.grid.device.type == "mps"
    assert torch.isfinite(kan.grid).all()
    y = kan(x)
    assert torch.isfinite(y).all()
    print("  [device] update_grid on MPS (CPU lstsq fallback): PASS")


def test_calibrate_grids_on_mps_model():
    """End-to-end calibration on an MPS-resident model: hooks fire, grids
    refit via the lstsq fallback, flags restored."""
    if not torch.backends.mps.is_available():
        print("  [device] MPS not available: SKIP")
        return
    model = MRKAN([4, 8, 1], [2, 2, 2], device=torch.device("mps"))
    model.calibrate_grids([torch.rand(2, 6, 4) for _ in range(2)], n_batches=2)
    for _, p in model.named_parameters():
        assert torch.isfinite(p).all()
    out = model(torch.rand(2, 5, 4, device="mps"))
    assert torch.isfinite(out).all()
    print("  [device] calibrate_grids on MPS model: PASS")


def main():
    print("=" * 70)
    print("Device resolution tests")
    print("=" * 70)
    tests = [
        test_default_device_preference_order,
        test_models_construct_and_run_on_default_device,
        test_update_grid_works_on_mps,
        test_calibrate_grids_on_mps_model,
    ]
    for t in tests:
        print()
        t()
    print("\n" + "=" * 70)
    print("All device resolution tests passed.")
    print("=" * 70)


if __name__ == "__main__":
    main()
