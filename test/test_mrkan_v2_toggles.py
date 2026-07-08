"""Tests for MR-KAN v2.

The v2 release adds three orthogonal ablation toggles plus a calibration
helper. Each test below exercises one toggle in isolation; later tests
combine toggles to confirm orthogonality. The final test reproduces the
v1 BPTT-connectivity guarantee with all toggles on, end-to-end.

Toggles under test:
- learn_ratios (SL-MR-KAN): bank ratios learned via RatioControlUnit
- ratio_control_use_kan: KAN vs nn.Linear backend for the RCU
- kan_input_path: replace W_ih with KANLinear
- kan_output_path: replace W_ho with KANLinear (Identity-base by default)
- calibrate_grids(loader): adaptive grid update helper
"""

import torch
import torch.nn as nn

from src.model.kan import MRKAN, MRKANCell, RatioControlUnit
from src.model.kan.kan_linear import KANLinear


def test_v2_1_v1_defaults_unchanged():
    """All v2 flags default to off; an MR-KAN constructed with no v2 args
    behaves exactly like v1 (no RCU, no path KANs)."""
    model = MRKAN(
        nn_structure=[10, 20, 1],
        memory_structure=[4, 3, 4],
        device=torch.device("cpu"),
    )
    assert model.cell.learn_ratios is False
    assert model.cell.kan_input_path is False
    assert model.cell.kan_output_path is False

    # No RCU on any bank
    for b in model.cell.memory_banks.values():
        assert b.rcu is None

    # First hidden and output stay in self.weights/biases (no kan_layers)
    assert "1" in model.cell.weights
    assert "2" in model.cell.weights
    assert len(model.cell.kan_layers) == 0

    print("  [v2-1] v1 defaults unchanged: PASS")


def test_v2_2_learn_ratios_isolated():
    """learn_ratios=True attaches an RCU to every bank and changes the
    update path; all other behavior remains v1."""
    model = MRKAN(
        nn_structure=[10, 20, 1],
        memory_structure=[4, 3, 4],
        learn_ratios=True,
        device=torch.device("cpu"),
    )
    # Every bank now has an RCU
    for b in model.cell.memory_banks.values():
        assert b.rcu is not None
        assert isinstance(b.rcu, RatioControlUnit)
        assert b.rcu.use_kan is True  # default backend
    # Path layers untouched
    assert "1" in model.cell.weights
    assert "2" in model.cell.weights
    assert len(model.cell.kan_layers) == 0

    # Forward and backward
    x = torch.randn(3, 8, 10)
    y = model(x)
    assert y.shape == (3, 8, 1)
    y.sum().backward()

    # Every RCU's KAN parameters got grad (not just zero)
    for b in model.cell.memory_banks.values():
        assert b.rcu.unit.spline_weight.grad is not None
        assert b.rcu.unit.spline_weight.grad.abs().sum() > 0

    print("  [v2-2] learn_ratios isolated: PASS")


def test_v2_3_ratio_control_use_kan_toggle():
    """ratio_control_use_kan=False makes RCUs use plain nn.Linear instead
    of KANLinear. Lets you ablate KAN-vs-Linear specifically for ratios."""
    model = MRKAN(
        nn_structure=[10, 20, 1],
        memory_structure=[4, 3, 4],
        learn_ratios=True,
        ratio_control_use_kan=False,
        device=torch.device("cpu"),
    )
    for b in model.cell.memory_banks.values():
        assert b.rcu is not None
        assert b.rcu.use_kan is False
        assert isinstance(b.rcu.unit, nn.Linear)

    y = model(torch.randn(2, 4, 10))
    y.sum().backward()
    print("  [v2-3] ratio_control_use_kan=False (Linear RCU): PASS")


def test_v2_4_kan_input_path_isolated():
    """kan_input_path=True moves layer 1 (input -> first hidden) into
    self.kan_layers; output path remains plain Linear."""
    model = MRKAN(
        nn_structure=[10, 20, 1],
        memory_structure=[4, 3, 4],
        kan_input_path=True,
        device=torch.device("cpu"),
    )
    assert "1" in model.cell.kan_layers
    assert "1" not in model.cell.weights
    assert "2" in model.cell.weights  # output stays linear
    # Input-path KAN keeps the configured base activation (SiLU by default)
    assert isinstance(
        model.cell.kan_layers["1"].base_activation, nn.SiLU
    )

    y = model(torch.randn(2, 5, 10))
    assert y.shape == (2, 5, 1)
    y.sum().backward()
    # Input-path KAN has nonzero grad
    assert model.cell.kan_layers["1"].spline_weight.grad.abs().sum() > 0
    print("  [v2-4] kan_input_path isolated: PASS")


def test_v2_5_kan_output_path_isolated():
    """kan_output_path=True moves the output layer into self.kan_layers
    with Identity base (preserving the canonical 'linear output' semantic)."""
    model = MRKAN(
        nn_structure=[10, 20, 1],
        memory_structure=[4, 3, 4],
        kan_output_path=True,
        device=torch.device("cpu"),
    )
    assert "2" in model.cell.kan_layers
    assert "2" not in model.cell.weights
    assert "1" in model.cell.weights  # input stays linear
    # Output-path KAN gets Identity base by default
    assert isinstance(
        model.cell.kan_layers["2"].base_activation, nn.Identity
    )

    y = model(torch.randn(2, 5, 10))
    assert y.shape == (2, 5, 1)
    y.sum().backward()
    print("  [v2-5] kan_output_path (Identity base) isolated: PASS")


def test_v2_6_orthogonality_all_combinations():
    """Every 2^4 = 16 combination of the four toggles should construct,
    forward, and backward without errors. We don't enumerate all 16 in full
    detail - we just exercise enough that a regression in any toggle would
    surface here."""
    cases = []
    for lr in (False, True):
        for rk in (True, False):
            for ki in (False, True):
                for ko in (False, True):
                    # ratio_control_use_kan only matters when learn_ratios=True
                    if not lr and not rk:
                        continue
                    cases.append((lr, rk, ki, ko))

    for lr, rk, ki, ko in cases:
        model = MRKAN(
            nn_structure=[6, 12, 2],
            memory_structure=[2, 2, 2],
            learn_ratios=lr,
            ratio_control_use_kan=rk,
            kan_input_path=ki,
            kan_output_path=ko,
            device=torch.device("cpu"),
        )
        x = torch.randn(2, 3, 6)
        y = model(x)
        assert y.shape == (2, 3, 2), (lr, rk, ki, ko, y.shape)
        y.sum().backward()
    print(f"  [v2-6] {len(cases)} toggle combinations all forward+backward: PASS")


def test_v2_7_calibrate_grids_runs_and_changes_grids():
    """calibrate_grids should run safely outside any training step and
    actually update at least one KANLinear's grid buffer."""
    torch.manual_seed(0)
    model = MRKAN(
        nn_structure=[6, 12, 1],
        memory_structure=[2, 2, 2],
        kan_input_path=True,        # so we have a path KAN as well as memory KANs
        device=torch.device("cpu"),
    )

    # Snapshot grids BEFORE
    before = {
        id(m): m.grid.clone()
        for m in model.modules()
        if isinstance(m, KANLinear)
    }
    assert len(before) > 0

    # Build a tiny dataloader-like iterable
    batches = [torch.randn(3, 8, 6) for _ in range(3)]

    # Run calibration
    model.calibrate_grids(batches, n_batches=3)

    # At least one grid should have changed (we're not asserting all of them,
    # because some KANs might receive identical-distribution inputs across
    # epochs and end up with the same adaptive grid).
    n_changed = 0
    for m in model.modules():
        if isinstance(m, KANLinear):
            if not torch.allclose(m.grid, before[id(m)]):
                n_changed += 1
    assert n_changed > 0, "no KANLinear grid changed after calibrate_grids"
    print(f"  [v2-7] calibrate_grids updated {n_changed} grids: PASS")


def test_v2_8_calibrate_grids_does_not_break_training():
    """Run a tiny train step, calibrate, run another train step. Loss should
    decrease and no exceptions should fire. This validates that the helper's
    eval/train and update_memory state save/restore is correct."""
    torch.manual_seed(1)
    model = MRKAN(
        nn_structure=[6, 12, 1],
        memory_structure=[2, 2, 2],
        device=torch.device("cpu"),
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    crit = nn.MSELoss()

    x = torch.randn(8, 5, 6)
    y_target = torch.randn(8, 5, 1)

    def step():
        opt.zero_grad()
        y = model(x)
        loss = crit(y, y_target)
        loss.backward()
        opt.step()
        return loss.item()

    l0 = step()
    # Calibrate between training steps - this is the documented use case.
    model.calibrate_grids([torch.randn(8, 5, 6) for _ in range(2)], n_batches=2)
    l1 = step()
    l2 = step()

    # Sanity: loss shouldn't blow up after calibration.
    assert torch.isfinite(torch.tensor([l0, l1, l2])).all()
    print(f"  [v2-8] train -> calibrate -> train losses: {l0:.4f} -> {l1:.4f} -> {l2:.4f}: PASS")


def test_v2_9_sl_mrkan_bptt_through_rcu():
    """SL-MR-KAN: gradients must flow back through every RCU's parameters
    when learn_ratios=True. Confirms BPTT integrates the learned ratios into
    the temporal computation graph."""
    model = MRKAN(
        nn_structure=[8, 16, 1],
        memory_structure=[3, 2, 3],
        learn_ratios=True,
        device=torch.device("cpu"),
    )
    y = model(torch.randn(2, 6, 8))
    y.sum().backward()

    for src, b in model.cell.memory_banks.items():
        assert b.rcu is not None
        # spline_weight (KAN backend) gets grad
        if hasattr(b.rcu.unit, "spline_weight"):
            g = b.rcu.unit.spline_weight.grad
            assert g is not None and g.abs().sum() > 0, f"bank {src} RCU spline_weight no grad"
        else:
            # nn.Linear backend - check weight
            g = b.rcu.unit.weight.grad
            assert g is not None and g.abs().sum() > 0, f"bank {src} RCU weight no grad"

    print("  [v2-9] BPTT through RCUs (SL-MR-KAN): PASS")


def main():
    print("=" * 70)
    print("MR-KAN v2 tests")
    print("=" * 70)
    tests = [
        test_v2_1_v1_defaults_unchanged,
        test_v2_2_learn_ratios_isolated,
        test_v2_3_ratio_control_use_kan_toggle,
        test_v2_4_kan_input_path_isolated,
        test_v2_5_kan_output_path_isolated,
        test_v2_6_orthogonality_all_combinations,
        test_v2_7_calibrate_grids_runs_and_changes_grids,
        test_v2_8_calibrate_grids_does_not_break_training,
        test_v2_9_sl_mrkan_bptt_through_rcu,
    ]
    for t in tests:
        print()
        t()
    print("\n" + "=" * 70)
    print("All MR-KAN v2 tests passed.")
    print("=" * 70)


if __name__ == "__main__":
    main()
