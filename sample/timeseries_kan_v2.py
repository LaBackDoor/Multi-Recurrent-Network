"""
Train MR-KAN v2 on COVID-19 Time Series Data, with optional v2.4 pruning.

Extends timeseries_kan.py with:
  - v2.4 post-training spline-shape pruning (--prune_threshold)
  - Pruning stats report: items dropped, parameter compression ratio
  - Pruned model saved and evaluated separately on the test set
  - Calibrate-grids support (inherited from v2.3)

All v2 toggles default to off so v1 behavior is preserved without any flags.
Pruning is also off by default (--prune_threshold defaults to 1.0, which
never drops any pair since cosine similarity is bounded in [0, 1]).
"""

import argparse
from pathlib import Path

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


def _build_model(args, nn_structure, device):
    model = MRKAN(
        nn_structure=nn_structure,
        memory_structure=args.memory_structure,
        kan_grid_size=args.kan_grid_size,
        kan_spline_order=args.kan_spline_order,
        kan_use_layernorm=args.kan_use_layernorm,
        kan_enable_standalone_scale_spline=args.kan_enable_standalone_scale_spline,
        learn_ratios=args.learn_ratios,
        ratio_control_use_kan=not args.ratio_control_linear,
        kan_input_path=args.kan_input_path,
        kan_output_path=args.kan_output_path,
        device=device,
    ).to(device)

    for name, param in model.named_parameters():
        if ".memory_kans." in name or "layer_norm" in name:
            continue
        if name.endswith("biases.1"):
            continue  # first hidden bias keeps hidden_bias_init_value (thesis 0.5)
        if name.endswith(".weight") or "weights." in name:
            if param.dim() >= 2:
                nn.init.xavier_uniform_(param)
        elif name.endswith(".bias") or "biases." in name:
            nn.init.zeros_(param)

    return model


def _print_pruning_stats(stats):
    print(f"\n  Similarity fn : {stats.similarity_fn_name}")
    print(f"  Threshold     : {stats.threshold}")
    print(f"  Items dropped : {stats.items_dropped}")
    print(f"  Items kept    : {stats.items_kept}")
    print(f"  Params before : {stats.params_before:,}")
    print(f"  Params after  : {stats.params_after:,}")
    ratio = stats.params_after / stats.params_before if stats.params_before else 1.0
    print(f"  Compression   : {ratio:.3f}x  ({(1 - ratio) * 100:.1f}% reduction)")
    print()
    for (src, tgt), bank in stats.banks.items():
        if bank.dropped_indices:
            pairs_str = ", ".join(
                f"({i},{j}) sim={s:.3f}" for i, j, s in bank.triggering_pairs
            )
            print(
                f"  Bank ({src}->{tgt}): dropped items {bank.dropped_indices} "
                f"[{bank.original_K} -> {bank.surviving_K}] | pairs: {pairs_str}"
            )
        else:
            print(
                f"  Bank ({src}->{tgt}): no drops "
                f"[K={bank.original_K}, max_sim={float(bank.similarity_matrix.max()):.3f}]"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Train MR-KAN v2 on COVID-19 Time Series (+ optional v2.4 pruning)"
    )

    parser.add_argument("--csv_path", type=str, default="data/time_series/c19.csv")
    parser.add_argument("--country", type=str, default="Afghanistan")
    parser.add_argument("--window_size", type=int, default=14)
    parser.add_argument("--horizon", type=int, default=7)
    parser.add_argument(
        "--use_minmax",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use MinMaxScaler (pass --no-use_minmax for StandardScaler)",
    )

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
    )

    # MR-KAN v2 ablation toggles
    parser.add_argument(
        "--learn_ratios",
        action="store_true",
        default=False,
        help="v2: enable SL-MR-KAN learnable per-item layer-link ratios.",
    )
    parser.add_argument(
        "--ratio_control_linear",
        action="store_true",
        default=False,
        help="v2: use nn.Linear RCU instead of KANLinear (ablation for learn_ratios).",
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
        help="v2: replace W_ho (last hidden -> output) with KANLinear.",
    )
    parser.add_argument(
        "--calibrate_grids_every",
        type=int,
        default=0,
        help="v2: adapt KAN grids every N epochs. 0 = off.",
    )
    parser.add_argument(
        "--calibrate_n_batches",
        type=int,
        default=4,
        help="v2: batches used per calibration call.",
    )

    # v2.4 pruning
    parser.add_argument(
        "--prune_threshold",
        type=float,
        default=1.0,
        help="v2.4: cosine-similarity threshold for pruning redundant memory "
             "items. Pairs above this are collapsed (lower index kept). "
             "1.0 = no pruning (default). Try 0.95 for mild, 0.85 for aggressive.",
    )
    parser.add_argument(
        "--prune_n_samples",
        type=int,
        default=128,
        help="v2.4: reference samples per bank for similarity computation.",
    )
    parser.add_argument(
        "--prune_finetune_epochs",
        type=int,
        default=0,
        help="v2.4: fine-tune the pruned model for this many extra epochs "
             "before final evaluation. 0 = eval immediately after prune.",
    )

    parser.add_argument("--output_dir", type=str, default="data/time_series/")
    parser.add_argument(
        "--model_tag",
        type=str,
        default="mrkan_v2",
        help="File tag for saved model/plot.",
    )

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
    print("Creating MR-KAN v2 Model")
    print("=" * 70)

    nn_structure = [1] + args.hidden_sizes + [1]
    model = _build_model(args, nn_structure, device)

    print(f"Network structure : {nn_structure}")
    print(f"Memory structure  : {model.cell.memory_structure}")
    print(
        f"KAN               : grid_size={args.kan_grid_size}, "
        f"spline_order={args.kan_spline_order}, "
        f"use_layernorm={args.kan_use_layernorm}, "
        f"standalone_scale={args.kan_enable_standalone_scale_spline}"
    )
    print(
        f"v2 toggles        : learn_ratios={args.learn_ratios}, "
        f"ratio_control_use_kan={not args.ratio_control_linear}, "
        f"kan_input_path={args.kan_input_path}, "
        f"kan_output_path={args.kan_output_path}, "
        f"calibrate_grids_every={args.calibrate_grids_every}"
    )
    print(
        f"v2.4 pruning      : threshold={args.prune_threshold}, "
        f"n_samples={args.prune_n_samples}, "
        f"finetune_epochs={args.prune_finetune_epochs}"
    )
    print(f"Total parameters  : {sum(p.numel() for p in model.parameters()):,}")

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

        # Calibrate BEFORE validating: update_grid rewrites grids and refits
        # spline_weight in place, so the checkpoint below must hold exactly
        # the weights that produced val_loss.
        if (
            args.calibrate_grids_every > 0
            and (epoch + 1) % args.calibrate_grids_every == 0
            and (epoch + 1) < args.epochs
        ):
            print(f"  [calibrate_grids] running on {args.calibrate_n_batches} batches")
            model.calibrate_grids(train_loader, n_batches=args.calibrate_n_batches)

        val_loss = evaluate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

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
    print(f"\nTest Loss (full model): {test_loss:.6f}")

    # v2.4: post-training pruning
    will_prune = args.prune_threshold < 1.0
    if will_prune:
        print("\n" + "=" * 70)
        print(f"Pruning  (threshold={args.prune_threshold})")
        print("=" * 70)

        # Reference inputs come from real evolved memory states (collected by
        # rolling the model over training batches), not random samples - the
        # similarity metric is only meaningful on-distribution.
        pruned_model, stats = model.prune(
            threshold=args.prune_threshold,
            n_samples=args.prune_n_samples,
            dataloader=train_loader,
        )
        pruned_model = pruned_model.to(device)
        _print_pruning_stats(stats)

        pruned_test_loss = evaluate(pruned_model, test_loader, criterion, device)
        print(f"Test Loss (pruned, no finetune): {pruned_test_loss:.6f}")

        if args.prune_finetune_epochs > 0:
            print(f"\nFine-tuning pruned model for {args.prune_finetune_epochs} epochs...")
            ft_optimizer = torch.optim.Adam(pruned_model.parameters(), lr=args.lr * 0.1)
            ft_best_val = float("inf")
            ft_ckpt = out_dir / f"best_{args.model_tag}_pruned_model.pth"

            for ep in range(args.prune_finetune_epochs):
                ft_train = train_epoch(
                    pruned_model, train_loader, criterion, ft_optimizer, device,
                    gradient_norm=1.0,
                )
                ft_val = evaluate(pruned_model, val_loader, criterion, device)
                if ft_val < ft_best_val:
                    ft_best_val = ft_val
                    torch.save(pruned_model.state_dict(), ft_ckpt)
                if (ep + 1) % 10 == 0 or ep == 0:
                    print(
                        f"  FT Epoch [{ep+1:3d}/{args.prune_finetune_epochs}] | "
                        f"Train: {ft_train:.6f} | Val: {ft_val:.6f}"
                    )

            pruned_model.load_state_dict(torch.load(ft_ckpt))
            pruned_test_loss = evaluate(pruned_model, test_loader, criterion, device)
            print(f"\nTest Loss (pruned + finetuned): {pruned_test_loss:.6f}")
        else:
            ft_ckpt = out_dir / f"best_{args.model_tag}_pruned_model.pth"
            torch.save(pruned_model.state_dict(), ft_ckpt)

        print(f"\nPruned model saved to {ft_ckpt}")

        # plot pruned model predictions
        plot_results(
            train_losses,
            val_losses,
            pruned_model,
            train_dataset,
            val_dataset,
            test_dataset,
            scaler,
            device,
            str(out_dir / f"{args.model_tag}_pruned_training_results.png"),
            args.country,
        )

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
