"""Regression tests for the 2026-07 bug-investigation fixes.

Covers:
- B1  calibrate_grids must restore the caller's _update_memory_flag.
- B1b calibrate_grids must let memory evolve, so memory-KAN grids are
      fitted to the run-time state distribution, not the init distribution.
- B2  collect_memory_references + prune(dataloader=...) provide
      on-distribution reference inputs for spline-shape similarity.
- B3  update_grid must never produce duplicate knots (NaN forward) for
      (near-)constant features, at any input magnitude.
- B4  MRKAN.prune must carry every constructor arg (and the memory-update
      flag) onto the pruned model.
- N2/N3 device handling: path KANs follow the construction device, and
      runtime tensor creation follows .to(device) moves.
- N5  sample-script re-init helpers must not clobber the thesis-canonical
      memory init or the first hidden bias.
- V   memory_structure longer than nn_structure is rejected, not silently
      ignored.
"""

import argparse

import torch
import torch.nn as nn

from src.model.kan import MRKAN
from src.model.kan.kan_linear import KANLinear
from src.model.mrn import MRN


def test_b1_calibrate_grids_preserves_update_memory_flag():
    """The flag must come back exactly as the caller left it."""
    model = MRKAN([4, 8, 1], [2, 2, 2], device=torch.device("cpu"))
    batches = [torch.randn(2, 6, 4)]

    model.set_update_memory(False)
    model.calibrate_grids(batches, n_batches=1)
    assert model.cell._update_memory_flag is False, (
        "calibrate_grids re-enabled memory updates the caller had disabled"
    )

    model.set_update_memory(True)
    model.calibrate_grids(batches, n_batches=1)
    assert model.cell._update_memory_flag is True
    print("  [b1] calibrate_grids restores _update_memory_flag: PASS")


def test_b1b_calibration_sees_evolving_memory():
    """Memory KANs must be calibrated on evolved states, not the init value.

    With constant init at 0.5 and frozen memory, every calibration sample
    equals 0.5 exactly and the knot grid collapses to a hair around
    LN(const) = 0. After the fix the hooked inputs must vary and the active
    knot span must stay usable."""
    torch.manual_seed(1)
    model = MRKAN(
        [4, 8, 1], [2, 0, 0],
        init_memory_mode="constant", init_memory_value=0.5,
        device=torch.device("cpu"),
    )
    kan0 = model.cell.memory_banks["0"].memory_kans["1"][0]
    seen = []
    handle = kan0.register_forward_hook(
        lambda mod, inp, out: seen.append(inp[0].detach().clone())
    )
    model.calibrate_grids([torch.randn(3, 10, 4)], n_batches=1)
    handle.remove()

    collected = torch.cat(seen, 0)
    assert (collected != 0.5).any(), (
        "memory never evolved during calibration; grids were fitted to the "
        "init distribution"
    )
    active = kan0.grid[:, kan0.spline_order:-kan0.spline_order]
    span = (active[:, -1] - active[:, 0]).min().item()
    assert span > 0.1, f"active knot span collapsed to {span:.2e}"
    print(f"  [b1b] calibration memory evolves (active span {span:.3f}): PASS")


def test_b2_collect_memory_references():
    """collect_memory_references returns per-source-layer evolved states."""
    torch.manual_seed(0)
    model = MRKAN([4, 8, 1], [3, 2, 2], device=torch.device("cpu"))
    batches = [torch.rand(2, 12, 4) for _ in range(3)]

    prev_flag = False
    model.set_update_memory(prev_flag)
    refs = model.collect_memory_references(batches, n_batches=3)

    assert set(refs.keys()) == {0, 1, 2}, refs.keys()
    assert refs[0].shape[1] == 4 and refs[1].shape[1] == 8 and refs[2].shape[1] == 1
    for src, r in refs.items():
        assert r.dim() == 2 and r.shape[0] > 0
        assert torch.isfinite(r).all()
    # hidden-layer memory is a convex blend of sigmoid activations and the
    # rand init, so it must live in (0, 1) - unlike randn reference inputs
    assert refs[1].min() >= 0.0 and refs[1].max() <= 1.0
    # flag restored
    assert model.cell._update_memory_flag is prev_flag
    print("  [b2] collect_memory_references shapes/support/flag: PASS")


def test_b2_prune_accepts_dataloader():
    """prune(dataloader=...) collects real refs instead of random fallback."""
    torch.manual_seed(0)
    model = MRKAN([4, 8, 1], [3, 0, 0], device=torch.device("cpu"))
    batches = [torch.rand(2, 10, 4) for _ in range(2)]
    pruned, stats = model.prune(threshold=0.999, dataloader=batches)
    assert stats.params_after <= stats.params_before
    out = pruned(torch.rand(2, 5, 4))
    assert out.shape == (2, 5, 1)
    print("  [b2] prune(dataloader=...) end-to-end: PASS")


def test_b3_update_grid_constant_features_no_nan():
    """(Near-)constant calibration features must not yield duplicate knots,
    NaN forwards, or non-finite grads - including at raw data magnitudes
    where fp32 rounding used to collapse adjacent knots."""
    cases = [
        ("exactly-constant @2e4", torch.full((32, 2), 20000.0), 0.01),
        ("near-constant @3e4", torch.full((64, 2), 30000.0) + torch.randn(64, 2) * 1e-3, 0.01),
        ("constant, margin=0", torch.full((32, 2), 0.5), 0.0),
        ("constant @unit scale", torch.full((32, 2), 0.5), 0.01),
    ]
    for name, x, margin in cases:
        torch.manual_seed(0)
        kan = KANLinear(2, 3, use_layernorm=False)
        kan.update_grid(x, margin=margin)
        spacing = kan.grid[:, 1:] - kan.grid[:, :-1]
        assert (spacing > 0).all(), f"{name}: {int((spacing <= 0).sum())} collapsed knot intervals"
        y = kan(x)
        assert torch.isfinite(y).all(), f"{name}: forward produced non-finite values"
        x2 = x.clone().requires_grad_(True)
        kan(x2).sum().backward()
        assert torch.isfinite(x2.grad).all(), f"{name}: non-finite input grads"
    print("  [b3] update_grid on (near-)constant features stays finite: PASS")


def test_b3_update_grid_normal_data_unchanged():
    """The knot-spacing guard must not distort grids fitted to healthy data."""
    torch.manual_seed(0)
    kan = KANLinear(3, 2, use_layernorm=False)
    x = torch.rand(256, 3) * 1.6 - 0.8
    kan.update_grid(x)
    active = kan.grid[:, kan.spline_order:-kan.spline_order]
    assert (active[:, 0] <= x.min(dim=0).values + 0.05).all()
    assert (active[:, -1] >= x.max(dim=0).values - 0.05).all()
    y = kan(x)
    assert torch.isfinite(y).all()
    print("  [b3] update_grid on healthy data keeps adaptive span: PASS")


def test_b4_prune_preserves_constructor_config():
    """Every constructor arg and the update flag must carry to the pruned model."""
    torch.manual_seed(0)
    model = MRKAN(
        [4, 8, 1], [3, 0, 0],
        weight_init_range=0.5,
        hidden_bias_init_value=None,
        init_memory_mode="constant",
        init_memory_value=0.25,
        device=torch.device("cpu"),
    )
    model.set_update_memory(False)
    pruned, _ = model.prune(threshold=1.0)

    assert pruned.cell.init_memory_mode == "constant"
    assert pruned.cell.init_memory_value == 0.25
    assert pruned.cell.weight_init_range == 0.5
    assert pruned.cell.hidden_bias_init_value is None
    assert pruned.cell._update_memory_flag is False
    # state built from the pruned model must honor the carried config
    state = pruned.init_state(2)
    assert torch.allclose(state.memory_banks[0], torch.full_like(state.memory_banks[0], 0.25))
    print("  [b4] prune carries constructor config + update flag: PASS")


def test_b4_prune_roundtrip_reconstruction():
    """Pruning the pruned model (attribute-driven reconstruction) must not
    silently revert config - the exact failure mode of the original bug."""
    torch.manual_seed(0)
    model = MRKAN(
        [4, 8, 1], [3, 2, 2],
        init_memory_mode="constant", init_memory_value=0.1,
        device=torch.device("cpu"),
    )
    once, _ = model.prune(threshold=1.0)
    twice, _ = once.prune(threshold=1.0)
    assert twice.cell.init_memory_mode == "constant"
    assert twice.cell.init_memory_value == 0.1
    print("  [b4] double-prune keeps config: PASS")


def test_n2_n3_device_movement():
    """Path KANs follow the construction device; runtime tensor creation
    follows .to(device). MPS stands in for CUDA on this machine."""
    if not (torch.backends.mps.is_available() or torch.cuda.is_available()):
        print("  [n2/n3] no accelerator available: SKIP")
        return
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cuda")

    # N2: constructing directly on the accelerator with path KANs enabled
    m = MRKAN([4, 8, 1], [2, 0, 0], kan_input_path=True, device=dev)
    devices = {p.device.type for p in m.parameters()}
    assert devices == {dev.type}, f"parameters scattered across devices: {devices}"
    out = m(torch.randn(2, 5, 4, device=dev))
    assert out.shape == (2, 5, 1)

    # N3: cpu-construct then .to(accelerator)
    m2 = MRKAN([4, 8, 1], [2, 2, 2], device=torch.device("cpu")).to(dev)
    state = m2.init_state(2)
    assert all(v.device.type == dev.type for v in state.memory_banks.values())
    out2 = m2(torch.randn(2, 5, 4, device=dev))
    assert out2.shape == (2, 5, 1)

    # same for the MRN baseline
    mrn = MRN([4, 8, 1], [2, 2, 2], device=torch.device("cpu")).to(dev)
    out3 = mrn(torch.randn(2, 5, 4, device=dev))
    assert out3.shape == (2, 5, 1)
    print(f"  [n2/n3] device movement on {dev.type}: PASS")


def test_n5_mrkan_build_model_preserves_thesis_init():
    """The MR-KAN sample's re-init must keep the 0.5 first-hidden bias and
    leave memory KANs alone."""
    from sample.timeseries_kan_v2 import _build_model

    args = argparse.Namespace(
        memory_structure=[2, 2, 2],
        kan_grid_size=3,
        kan_spline_order=3,
        kan_use_layernorm=True,
        kan_enable_standalone_scale_spline=False,
        learn_ratios=False,
        ratio_control_linear=False,
        kan_input_path=False,
        kan_output_path=False,
    )
    model = _build_model(args, [1, 8, 1], torch.device("cpu"))
    b1 = model.cell.biases["1"]
    assert torch.allclose(b1, torch.full_like(b1, 0.5)), (
        "first hidden bias no longer matches hidden_bias_init_value=0.5"
    )
    # feedforward weights re-initialized (xavier scale >> 0.01 uniform)
    assert model.cell.weights["1"].abs().max() > 0.02
    print("  [n5] MR-KAN sample init keeps thesis hidden bias: PASS")


def test_n5_mrn_baseline_init_scoped_to_feedforward():
    """The MRN sample's re-init must not clobber memory projections
    (thesis: uniform(-0.01, 0.01)) or the 0.5 first-hidden bias."""
    from sample.timeseries import init_baseline_weights

    torch.manual_seed(0)
    model = MRN([1, 8, 1], [2, 2, 2], device=torch.device("cpu"))
    init_baseline_weights(model)

    for name, p in model.named_parameters():
        if "memory_weights" in name:
            assert p.abs().max() <= 0.01 + 1e-6, (
                f"{name} was re-initialized off the thesis range"
            )
    b1 = model.cell.biases["1"]
    assert torch.allclose(b1, torch.full_like(b1, 0.5))
    assert model.cell.weights["1"].abs().max() > 0.02  # xavier applied
    assert model.cell.biases["2"].abs().max() == 0.0   # output bias zeroed
    print("  [n5] MRN sample init scoped to feedforward paths: PASS")


def test_noop_prune_is_identity_for_sl_mrkan():
    """A prune that drops nothing must not change the model - previously it
    still rebuilt every RCU without LayerNorm (params and outputs shifted)."""
    torch.manual_seed(0)
    model = MRKAN(
        [4, 8, 1], [3, 2, 2],
        learn_ratios=True,
        device=torch.device("cpu"),
    )
    pruned, stats = model.prune(threshold=1.0)
    assert stats.items_dropped == 0

    n_before = sum(p.numel() for p in model.parameters())
    n_after = sum(p.numel() for p in pruned.parameters())
    assert n_before == n_after, (
        f"no-drop prune changed param count: {n_before} -> {n_after} "
        "(RCU LayerNorm stripped despite nothing being pruned)"
    )

    torch.manual_seed(42)
    x = torch.rand(2, 6, 4)
    state = model.init_state(2)
    out_orig = model(x, states=state.clone())
    out_pruned = pruned(x, states=state.clone())
    diff = (out_orig - out_pruned).abs().max().item()
    assert diff < 1e-6, f"no-drop prune changed outputs by {diff}"
    print("  [noop] threshold=1.0 prune is exact identity (SL-MR-KAN): PASS")


def test_v_memory_structure_length_validated():
    """Extra memory_structure entries used to be silently ignored."""
    for ctor in (
        lambda: MRKAN([4, 8, 1], [1, 1, 1, 4], device=torch.device("cpu")),
        lambda: MRN([4, 8, 1], [1, 1, 1, 4], device=torch.device("cpu")),
    ):
        try:
            ctor()
        except ValueError:
            continue
        raise AssertionError("expected ValueError for oversized memory_structure")
    print("  [v] oversized memory_structure rejected: PASS")


def main():
    print("=" * 70)
    print("MR-KAN bug-fix regression tests")
    print("=" * 70)
    tests = [
        test_b1_calibrate_grids_preserves_update_memory_flag,
        test_b1b_calibration_sees_evolving_memory,
        test_b2_collect_memory_references,
        test_b2_prune_accepts_dataloader,
        test_b3_update_grid_constant_features_no_nan,
        test_b3_update_grid_normal_data_unchanged,
        test_b4_prune_preserves_constructor_config,
        test_b4_prune_roundtrip_reconstruction,
        test_n2_n3_device_movement,
        test_n5_mrkan_build_model_preserves_thesis_init,
        test_n5_mrn_baseline_init_scoped_to_feedforward,
        test_noop_prune_is_identity_for_sl_mrkan,
        test_v_memory_structure_length_validated,
    ]
    for t in tests:
        print()
        t()
    print("\n" + "=" * 70)
    print("All bug-fix regression tests passed.")
    print("=" * 70)


if __name__ == "__main__":
    main()
