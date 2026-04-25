"""
Train MR-KAN on COVID-19 Time Series Data.

Thin adapter over sample/timeseries.py: reuses the MRN pipeline's data loading,
training loop, and plotting utilities, and swaps the model class for MRKAN.
Exposes the MR-KAN-specific knobs (grid_size, spline_order, use_layernorm, etc.)
so ablations against the MRN baseline are a CLI flag away.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sample.timeseries import (
    evaluate,
    load_data,
    plot_results,
    prepare_data,
    train_epoch,
)
from src.model.kan import MRKAN


def main():
    parser = argparse.ArgumentParser(description="Train MR-KAN on COVID-19 Time Series")

    parser.add_argument("--csv_path", type=str, default="data/time_series/c19.csv")
    parser.add_argument("--country", type=str, default="Afghanistan")
    parser.add_argument("--window_size", type=int, default=14)
    parser.add_argument("--horizon", type=int, default=7)
    parser.add_argument("--use_minmax", action="store_true", default=True)

    parser.add_argument("--hidden_sizes", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--memory_structure", type=int, nargs="+", default=[4, 3, 2, 4])

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--val_size", type=float, default=0.15)

    # MR-KAN v1 args
    parser.add_argument("--kan_grid_size", type=int, default=3)
    parser.add_argument("--kan_spline_order", type=int, default=3)
    parser.add_argument("--kan_use_layernorm", action="store_true", default=True)
    parser.add_argument("--kan_no_layernorm", dest="kan_use_layernorm", action="store_false")
    parser.add_argument(
        "--kan_enable_standalone_scale_spline",
        action="store_true",
        default=False,
        help="Adds a per-edge learnable scaler to the spline branch (default off for tighter cells).",
    )

    # MR-KAN v2 ablation toggles - all default off so behavior matches v1
    parser.add_argument(
        "--learn_ratios",
        action="store_true",
        default=False,
        help="v2: enable SL-MR-KAN. Each bank gets a learned RatioControlUnit "
             "instead of fixed (i+1)/K ratios.",
    )
    parser.add_argument(
        "--ratio_control_linear",
        action="store_true",
        default=False,
        help="v2: when --learn_ratios is on, use a plain nn.Linear RCU "
             "instead of KANLinear. Lets you ablate KAN-vs-Linear for ratios "
             "specifically.",
    )
    parser.add_argument(
        "--kan_input_path",
        action="store_true",
        default=False,
        help="v2: replace W_ih (input -> first hidden) with KANLinear.",
    )
    parser.add_argument(
        "--kan_output_path",
        action="store_true",
        default=False,
        help="v2: replace W_ho (last hidden -> output) with KANLinear "
             "(Identity base preserves linear-output semantics).",
    )
    parser.add_argument(
        "--calibrate_grids_every",
        type=int,
        default=0,
        help="v2: if > 0, run model.calibrate_grids on the train loader every "
             "N epochs to adapt KAN grids to the input distribution. 0 = off.",
    )
    parser.add_argument(
        "--calibrate_n_batches",
        type=int,
        default=4,
        help="v2: number of batches to use per calibration call.",
    )

    parser.add_argument("--output_dir", type=str, default="data/time_series/")
    parser.add_argument("--model_tag", type=str, default="mrkan",
                        help="File tag for saved model/plot; distinct from MRN's 'mrn'.")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("=" * 70)
    print(f"Loading {args.country} COVID-19 Data")
    print("=" * 70)
    case_data = load_data(args.csv_path)

    print("\n" + "=" * 70)
    print("Preparing Data")
    print("=" * 70)
    train_dataset, val_dataset, test_dataset, scaler = prepare_data(
        case_data,
        window_size=args.window_size,
        horizon=args.horizon,
        test_size=args.test_size,
        val_size=args.val_size,
        use_minmax=args.use_minmax,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print("\n" + "=" * 70)
    print("Creating MR-KAN Model")
    print("=" * 70)

    nn_structure = [1] + args.hidden_sizes + [1]

    model = MRKAN(
        nn_structure=nn_structure,
        memory_structure=args.memory_structure,
        kan_grid_size=args.kan_grid_size,
        kan_spline_order=args.kan_spline_order,
        kan_use_layernorm=args.kan_use_layernorm,
        kan_enable_standalone_scale_spline=args.kan_enable_standalone_scale_spline,
        # v2 ablation toggles
        learn_ratios=args.learn_ratios,
        ratio_control_use_kan=not args.ratio_control_linear,
        kan_input_path=args.kan_input_path,
        kan_output_path=args.kan_output_path,
        device=device,
    ).to(device)

    # Reinitialize feedforward weights with xavier to match the MRN baseline script.
    # KANLinear parameters (base_weight, spline_weight, spline_scaler) are left
    # at efficient-kan's carefully-tuned init; overriding them with xavier would
    # defeat the near-zero spline initialization that stabilises early training.
    for name, param in model.named_parameters():
        if ".memory_kans." in name:
            continue  # leave KAN init alone
        if "layer_norm" in name:
            continue
        if name.endswith(".weight") or "weights." in name:
            if param.dim() >= 2:
                nn.init.xavier_uniform_(param)
        elif name.endswith(".bias") or "biases." in name:
            nn.init.zeros_(param)

    print(f"Network structure: {nn_structure}")
    print(f"Memory structure: {model.cell.memory_structure}")
    print(f"KAN grid_size={args.kan_grid_size}, spline_order={args.kan_spline_order}, "
          f"use_layernorm={args.kan_use_layernorm}, "
          f"standalone_scale_spline={args.kan_enable_standalone_scale_spline}")
    print(f"v2 toggles: learn_ratios={args.learn_ratios}, "
          f"ratio_control_use_kan={not args.ratio_control_linear}, "
          f"kan_input_path={args.kan_input_path}, "
          f"kan_output_path={args.kan_output_path}, "
          f"calibrate_grids_every={args.calibrate_grids_every}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6
    )

    print("\n" + "=" * 70)
    print("Training")
    print("=" * 70)

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"best_{args.model_tag}_model.pth"

    for epoch in range(args.epochs):
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, gradient_norm=1.0
        )
        val_loss = evaluate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        # v2: optional adaptive grid update between epochs.
        if (
            args.calibrate_grids_every > 0
            and (epoch + 1) % args.calibrate_grids_every == 0
            and (epoch + 1) < args.epochs
        ):
            print(f"  [calibrate_grids] running on {args.calibrate_n_batches} batches")
            model.calibrate_grids(train_loader, n_batches=args.calibrate_n_batches)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_path)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch [{epoch+1:3d}/{args.epochs}] | "
                f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}"
            )

    model.load_state_dict(torch.load(ckpt_path))
    test_loss = evaluate(model, test_loader, criterion, device)
    print(f"\nFinal Test Loss: {test_loss:.6f}")

    print("\n" + "=" * 70)
    print("Generating Plots")
    print("=" * 70)
    plot_results(
        train_losses,
        val_losses,
        model,
        train_dataset,
        val_dataset,
        test_dataset,
        scaler,
        device,
        str(out_dir / f"{args.model_tag}_training_results.png"),
        args.country,
    )

    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
