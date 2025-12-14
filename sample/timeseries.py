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
        """
        Create input-output sequences using a sliding window.
        - Input: window_size timesteps starting at position i
        - Output: window_size timesteps starting at position i+horizon

        This means at each timestep in the input, we predict horizon steps ahead.
        """
        x, y = [], []

        for i in range(len(self.data) - self.window_size - self.horizon + 1):
            # Input: window_size timesteps [i, i+window_size)
            x.append(self.data[i : i + self.window_size])

            # Output: window_size timesteps starting at i+horizon
            # [i+horizon, i+horizon+window_size)
            y.append(self.data[i + self.horizon : i + self.horizon + self.window_size])

        return np.array(x), np.array(y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.FloatTensor(self.X[idx]).unsqueeze(-1),  # [window_size, 1]
            torch.FloatTensor(self.y[idx]),  # [window_size] - targets at each timestep
        )


def load_data(csv_path: str, country: str = "Afghanistan") -> np.ndarray:
    """
    Load COVID-19 data for the country from CSV.

    Args:
        csv_path: Path to the COVID-19 CSV file
        country: country to filter out

    Returns:
        1D numpy array of case counts over time
    """
    df = pd.read_csv(csv_path)

    # Filter Afghanistan row
    data_row = df[df["Country/Region"] == country].iloc[0]

    # Extract case data (skip first 4 columns: Province/State, Country/Region, Lat, Long)
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
    """
    Prepare data with train/val/test split and normalization.

    Args:
        data: Raw time series data
        window_size: Input sequence length
        horizon: Output sequence length
        test_size: Fraction of data for test set
        val_size: Fraction of train data for validation
        use_minmax: If True, use MinMaxScaler; if False, use StandardScaler

    Returns:
        train_dataset, val_dataset, test_dataset, scaler
    """
    # Split data temporally (test is last 15% of data)
    test_split_idx = int(len(data) * (1 - test_size))
    train_val_data = data[:test_split_idx]
    test_data = data[test_split_idx:]

    # Split train into train and validation
    val_split_idx = int(len(train_val_data) * (1 - val_size))
    train_data = train_val_data[:val_split_idx]
    val_data = train_val_data[val_split_idx:]

    print(f"\nData split:")
    print(f"  Train: {len(train_data)} days")
    print(f"  Val:   {len(val_data)} days")
    print(f"  Test:  {len(test_data)} days")

    # Scale data (fit on train only)
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

    # Create datasets
    train_dataset = TimeSeriesDataset(train_data_scaled, window_size, horizon)
    val_dataset = TimeSeriesDataset(val_data_scaled, window_size, horizon)
    test_dataset = TimeSeriesDataset(test_data_scaled, window_size, horizon)

    print(f"\nDataset sizes (sequences):")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Val:   {len(val_dataset)}")
    print(f"  Test:  {len(test_dataset)}")

    return train_dataset, val_dataset, test_dataset, scaler


def train_epoch(
    model: MRN,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_norm: float,
) -> float:
    """
    Train for one epoch.

    - Process each timestep in the input sequence
    - At each timestep I, predict the value at timestep i+horizon
    - Compute loss over ALL timesteps
    """
    model.train()
    total_loss = 0.0

    for batch_x, batch_y in data_loader:
        batch_x = batch_x.to(device)  # [batch_size, window_size, 1]
        batch_y = batch_y.to(device)  # [batch_size, window_size]

        optimizer.zero_grad()

        # Reset memory at the start of each sequence
        model.reset_memory()

        # Process the full input sequence - get output at EACH timestep
        outputs = model(batch_x, return_sequences=True)  # [batch_size, window_size, 1]

        # Remove the feature dimension
        outputs = outputs.squeeze(-1)  # [batch_size, window_size]

        # Compute loss over ALL timesteps
        # outputs[: i] should match batch_y[: i]
        # This means: at input timestep i, we predict the value horizon steps ahead
        loss = criterion(outputs, batch_y)
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_norm)

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)


def evaluate(
    model: MRN, data_loader: DataLoader, criterion: nn.Module, device: torch.device
) -> float:
    """Evaluate model on validation/test set."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # Reset memory at the start of each sequence
            model.reset_memory()

            # Process input sequence - get outputs at all timesteps
            outputs = model(
                batch_x, return_sequences=True
            )  # [batch_size, window_size, 1]
            outputs = outputs.squeeze(-1)  # [batch_size, window_size]

            # Compute loss over all timesteps
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
    """
    Plot training curves and predictions.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Plot 1: Training and validation loss
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

    # Plot 2: Predictions vs. Actual (full-time series)
    model.eval()

    # Collect predictions for all datasets
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
                x = x.unsqueeze(0).to(device)  # [1, window_size, 1]

                # Reset memory for each sequence
                model.reset_memory()

                # Process input sequence through network
                outputs = model(x, return_sequences=True)  # [1, window_size, 1]

                # Take the LAST output (predicts horizon steps from last input timestep)
                prediction = outputs[0, -1, 0].item()

                all_predictions.append(prediction)
                # Target is also the last value (horizon steps ahead from last input)
                all_actuals.append(y[-1].item())

    # Inverse transform to original scale
    predictions_original = scaler.inverse_transform(
        np.array(all_predictions).reshape(-1, 1)
    ).flatten()
    actuals_original = scaler.inverse_transform(
        np.array(all_actuals).reshape(-1, 1)
    ).flatten()

    # Calculate split points
    train_end = len(train_dataset)
    val_end = train_end + len(val_dataset)

    # Plot predictions vs actuals
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

    # Add vertical lines to show train/val/test splits
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

    # Calculate metrics for each split
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

    # Add metrics text
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

    # Print final metrics
    print(f"\nFull Metrics:")
    print(f"  Train - RMSE: {train_rmse:.2f}, MAE: {train_mae:.2f}")
    print(f"  Val   - RMSE: {val_rmse:.2f}, MAE: {val_mae:.2f}")
    print(f"  Test  - RMSE: {test_rmse:.2f}, MAE: {test_mae:.2f}")


def main():

    parser = argparse.ArgumentParser(description="Train MRN on COVID-19 Time Series")

    # Data parameters
    parser.add_argument(
        "--csv_path",
        type=str,
        default="../data/time_series/c19.csv",
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
        action="store_true",
        default=True,
        help="Use MinMaxScaler instead of StandardScaler",
    )

    # Model parameters
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

    # Training parameters
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

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../data/time_series/",
        help="Directory to save results",
    )

    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print("=" * 70)
    print(f"Loading {args.country} COVID-19 Data")
    print("=" * 70)
    case_data = load_data(args.csv_path)

    # Prepare data
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

    # Create data_loaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Create an MRN model
    print("\n" + "=" * 70)
    print("Creating MRN Model")
    print("=" * 70)

    # Build network structure: [input_size] + hidden_sizes + [output_size]
    # Output size should be 1 (predict single next value)
    nn_structure = [1] + args.hidden_sizes + [1]

    model = MRN(
        nn_structure=nn_structure, memory_structure=args.memory_structure, device=device
    ).to(device)

    # Initialize weights with Xavier/Glorot initialization
    def init_weights(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Parameter):
            if hasattr(m, "weight"):
                nn.init.xavier_uniform_(m.weight)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.zeros_(m.bias)

    # Apply initialization to all parameters
    for name, param in model.named_parameters():
        if "weight" in name:
            nn.init.xavier_uniform_(param)
        elif "bias" in name:
            nn.init.zeros_(param)

    init_weights(model)

    print(f"Network structure: {nn_structure}")
    print(f"Memory structure: {model.cell.memory_structure}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Learning rate scheduler - reduce LR when validation loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6
    )

    # Training loop
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

        # Step the learning rate scheduler
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save the best model
            torch.save(model.state_dict(), Path(args.output_dir) / "best_mrn_model.pth")

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch [{epoch+1:3d}/{args.epochs}] | "
                f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}"
            )

    # Load the best model for final evaluation
    model.load_state_dict(torch.load(Path(args.output_dir) / "best_mrn_model.pth"))

    # Final test evaluation
    test_loss = evaluate(model, test_loader, criterion, device)
    print(f"\nFinal Test Loss: {test_loss:.6f}")

    # Plot results
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
