"""
Train Multi-Recurrent Neural Network (MRN) on COVID-19 Time Series Data
Focus: Afghanistan case data with sliding window approach
"""

import argparse
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import Dataset, DataLoader

from src.model.device import default_device
from src.model.mrn import MRN


class TimeSeriesDataset(Dataset):
    """
    Time series dataset with sliding window approach.

    - Input: window_size timesteps [i, i+window_size]
    - Output: window_size timesteps [i+horizon, i+horizon+window_size]

    This means at each input timestep, we predict horizon steps ahead from that point.
    Training uses ALL outputs from the sequence, not just the last one.

    Args:
        data: Normalized time series data (1D array)
        window_size: Number of past timesteps to use as input
        horizon: How many steps ahead to predict at each timestep
    """

    def __init__(self, data: np.ndarray, window_size: int, horizon: int):
        self.data = data
        self.window_size = window_size
        self.horizon = horizon
        self.X, self.y = self._create_sequences()

    def _create_sequences(self) -> Tuple[np.ndarray, np.ndarray]:
        x, y = [], []

        for i in range(len(self.data) - self.window_size - self.horizon + 1):
            x.append(self.data[i : i + self.window_size])
            y.append(self.data[i + self.horizon : i + self.horizon + self.window_size])

        return np.array(x), np.array(y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.FloatTensor(self.X[idx]).unsqueeze(-1),
            torch.FloatTensor(self.y[idx]),
        )


def load_data(csv_path: str, country: str = "Afghanistan") -> np.ndarray:
    """Load COVID-19 case counts for one country from the JHU CSV format."""
    df = pd.read_csv(csv_path)

    data_row = df[df["Country/Region"] == country].iloc[0]

    # Skip first 4 columns: Province/State, Country/Region, Lat, Long.
    case_data = data_row.iloc[4:].values.astype(float)

    print(f"Loaded {country} data: {len(case_data)} days")
    print(f"Date range: {df.columns[4]} to {df.columns[-1]}")
    print(f"Cases range: {case_data.min():.0f} to {case_data.max():.0f}")

    return case_data


def prepare_data(
    data: np.ndarray,
    window_size: int,
    horizon: int,
    test_size: float = 0.15,
    val_size: float = 0.15,
    use_minmax: bool = True,
) -> Tuple[
    TimeSeriesDataset,
    TimeSeriesDataset,
    TimeSeriesDataset,
    MinMaxScaler | StandardScaler,
]:
    """Split temporally into train/val/test and fit a scaler on train only."""
    test_split_idx = int(len(data) * (1 - test_size))
    train_val_data = data[:test_split_idx]
    test_data = data[test_split_idx:]

    val_split_idx = int(len(train_val_data) * (1 - val_size))
    train_data = train_val_data[:val_split_idx]
    val_data = train_val_data[val_split_idx:]

    print(f"\nData split:")
    print(f"  Train: {len(train_data)} days")
    print(f"  Val:   {len(val_data)} days")
    print(f"  Test:  {len(test_data)} days")

    if use_minmax:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler_name = "MinMax"
    else:
        scaler = StandardScaler()
        scaler_name = "Standard"

    train_data_scaled = scaler.fit_transform(train_data.reshape(-1, 1)).flatten()
    val_data_scaled = scaler.transform(val_data.reshape(-1, 1)).flatten()
    test_data_scaled = scaler.transform(test_data.reshape(-1, 1)).flatten()

    print(f"\n{scaler_name} Scaling:")
    if use_minmax:
        print(f"  Min: {scaler.data_min_[0]:.2f}")
        print(f"  Max: {scaler.data_max_[0]:.2f}")
    else:
        print(f"  Mean: {scaler.mean_[0]:.2f}")
        print(f"  Std:  {scaler.scale_[0]:.2f}")

    train_dataset = TimeSeriesDataset(train_data_scaled, window_size, horizon)
    val_dataset = TimeSeriesDataset(val_data_scaled, window_size, horizon)
    test_dataset = TimeSeriesDataset(test_data_scaled, window_size, horizon)

    print(f"\nDataset sizes (sequences):")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Val:   {len(val_dataset)}")
    print(f"  Test:  {len(test_dataset)}")

    return train_dataset, val_dataset, test_dataset, scaler


def init_baseline_weights(model: MRN) -> None:
    """Xavier/zeros re-init for the plain feedforward paths only.

    Scope matters: memory-projection weights keep the model's thesis-canonical
    uniform(-weight_init_range, weight_init_range) init, and the first hidden
    bias keeps hidden_bias_init_value (0.5). A broad "weight"/"bias" substring
    match would silently clobber both and break the documented v1 protocol.
    """
    for name, param in model.named_parameters():
        if "memory_weights" in name:
            continue
        if "weights." in name and param.dim() >= 2:
            nn.init.xavier_uniform_(param)
        elif "biases." in name and not name.endswith("biases.1"):
            nn.init.zeros_(param)


def train_epoch(
    model: MRN,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_norm: float,
) -> float:
    model.train()
    total_loss = 0.0

    for batch_x, batch_y in data_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()

        outputs = model(batch_x, return_sequences=True)
        outputs = outputs.squeeze(-1)

        loss = criterion(outputs, batch_y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_norm)

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)


def evaluate(
    model: MRN, data_loader: DataLoader, criterion: nn.Module, device: torch.device
) -> float:
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_x, return_sequences=True)
            outputs = outputs.squeeze(-1)

            loss = criterion(outputs, batch_y)
            total_loss += loss.item()

    return total_loss / len(data_loader)


def plot_results(
    train_losses: list,
    val_losses: list,
    model: MRN,
    train_dataset: TimeSeriesDataset,
    val_dataset: TimeSeriesDataset,
    test_dataset: TimeSeriesDataset,
    scaler: StandardScaler,
    device: torch.device,
    save_path: str,
    country: str,
):
    """Plot training curves and predictions."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    axes[0].plot(train_losses, label="Train Loss", linewidth=2)
    axes[0].plot(val_losses, label="Validation Loss", linewidth=2)
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Loss (MSE)", fontsize=12)
    axes[0].set_title(
        f"MRN Training Progress on {country} COVID-19 Data",
        fontsize=14,
        fontweight="bold",
    )
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    model.eval()

    all_datasets = [
        (train_dataset, "train"),
        (val_dataset, "val"),
        (test_dataset, "test"),
    ]

    all_predictions = []
    all_actuals = []

    with torch.no_grad():
        for dataset, split_name in all_datasets:
            for i in range(len(dataset)):
                x, y = dataset[i]
                x = x.unsqueeze(0).to(device)

                outputs = model(x, return_sequences=True)

                prediction = outputs[0, -1, 0].item()

                all_predictions.append(prediction)
                all_actuals.append(y[-1].item())

    predictions_original = scaler.inverse_transform(
        np.array(all_predictions).reshape(-1, 1)
    ).flatten()
    actuals_original = scaler.inverse_transform(
        np.array(all_actuals).reshape(-1, 1)
    ).flatten()

    train_end = len(train_dataset)
    val_end = train_end + len(val_dataset)

    time_steps = np.arange(len(actuals_original))
    axes[1].plot(
        time_steps,
        actuals_original,
        label="Actual Cases",
        linewidth=2,
        alpha=0.8,
        color="blue",
    )
    axes[1].plot(
        time_steps,
        predictions_original,
        label="Predicted Cases",
        linewidth=2,
        alpha=0.8,
        linestyle="--",
        color="orange",
    )

    axes[1].axvline(
        x=train_end,
        color="red",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label="Train/Val Split",
    )
    axes[1].axvline(
        x=val_end,
        color="green",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label="Val/Test Split",
    )

    axes[1].set_xlabel("Sequence Index", fontsize=12)
    axes[1].set_ylabel("COVID-19 Cases", fontsize=12)
    axes[1].set_title(
        "Full Time Series: Predictions vs Actual", fontsize=14, fontweight="bold"
    )
    axes[1].legend(fontsize=10, loc="upper right")
    axes[1].grid(True, alpha=0.3)

    train_pred = predictions_original[:train_end]
    train_actual = actuals_original[:train_end]
    val_pred = predictions_original[train_end:val_end]
    val_actual = actuals_original[train_end:val_end]
    test_pred = predictions_original[val_end:]
    test_actual = actuals_original[val_end:]

    train_rmse = np.sqrt(np.mean((train_pred - train_actual) ** 2))
    val_rmse = np.sqrt(np.mean((val_pred - val_actual) ** 2))
    test_rmse = np.sqrt(np.mean((test_pred - test_actual) ** 2))

    train_mae = np.mean(np.abs(train_pred - train_actual))
    val_mae = np.mean(np.abs(val_pred - val_actual))
    test_mae = np.mean(np.abs(test_pred - test_actual))

    metrics_text = (
        f"Train RMSE: {train_rmse:.2f} | MAE: {train_mae:.2f}\n"
        f"Val RMSE: {val_rmse:.2f} | MAE: {val_mae:.2f}\n"
        f"Test RMSE: {test_rmse:.2f} | MAE: {test_mae:.2f}"
    )
    axes[1].text(
        0.02,
        0.98,
        metrics_text,
        transform=axes[1].transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        fontsize=9,
        family="monospace",
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to: {save_path}")

    print(f"\nFull Metrics:")
    print(f"  Train - RMSE: {train_rmse:.2f}, MAE: {train_mae:.2f}")
    print(f"  Val   - RMSE: {val_rmse:.2f}, MAE: {val_mae:.2f}")
    print(f"  Test  - RMSE: {test_rmse:.2f}, MAE: {test_mae:.2f}")


def main():

    parser = argparse.ArgumentParser(description="Train MRN on COVID-19 Time Series")

    parser.add_argument(
        "--csv_path",
        type=str,
        default="data/time_series/c19.csv",
        help="Path to COVID-19 CSV file",
    )
    parser.add_argument("--country", type=str, default="Afghanistan", help="Country")
    parser.add_argument(
        "--window_size",
        type=int,
        default=14,
        help="Number of past days to use as input",
    )
    parser.add_argument(
        "--horizon", type=int, default=7, help="Number of future days to predict"
    )
    parser.add_argument(
        "--use_minmax",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use MinMaxScaler (pass --no-use_minmax for StandardScaler)",
    )

    parser.add_argument(
        "--hidden_sizes",
        type=int,
        nargs="+",
        default=[16, 32],
        help="Sizes of hidden layers",
    )
    parser.add_argument(
        "--memory_structure",
        type=int,
        nargs="+",
        default=[4, 3, 2, 4],
        help="Number of memory banks per layer",
    )

    parser.add_argument(
        "--epochs", type=int, default=200, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--test_size", type=float, default=0.15, help="Fraction of data for test set"
    )
    parser.add_argument(
        "--val_size",
        type=float,
        default=0.15,
        help="Fraction of train data for validation",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/time_series/",
        help="Directory to save results",
    )

    args = parser.parse_args()

    device = default_device()
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
    print("Creating MRN Model")
    print("=" * 70)

    nn_structure = [1] + args.hidden_sizes + [1]

    model = MRN(
        nn_structure=nn_structure, memory_structure=args.memory_structure, device=device
    ).to(device)

    init_baseline_weights(model)

    print(f"Network structure: {nn_structure}")
    print(f"Memory structure: {model.cell.memory_structure}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6
    )

    print("\n" + "=" * 70)
    print("Training")
    print("=" * 70)

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, gradient_norm=1.0
        )
        val_loss = evaluate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), Path(args.output_dir) / "best_mrn_model.pth")

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch [{epoch+1:3d}/{args.epochs}] | "
                f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}"
            )

    model.load_state_dict(torch.load(Path(args.output_dir) / "best_mrn_model.pth"))

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
        str(Path(args.output_dir) / "mrn_training_results.png"),
        args.country,
    )

    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
