"""
Ensemble of Multi-Recurrent Neural Networks (MRN) for Time Series Forecasting

Trains multiple MRN models with different architectures/initializations and combines
their predictions using various ensemble techniques.
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict, Union, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader

from sample.timeseries import TimeSeriesDataset
from src.model.mrn import MRN


def load_data(csv_path: str, country: str = "Afghanistan") -> np.ndarray:
    """Load COVID-19 data."""
    df = pd.read_csv(csv_path)
    data_row = df[df["Country/Region"] == country].iloc[0]
    case_data = data_row.iloc[4:].values.astype(float)

    print(f"Loaded {country} data: {len(case_data)} days")
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
    """Prepare train/val/test datasets."""
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

    scaler = MinMaxScaler(feature_range=(0, 1)) if use_minmax else StandardScaler()

    train_data_scaled = scaler.fit_transform(train_data.reshape(-1, 1)).flatten()
    val_data_scaled = scaler.transform(val_data.reshape(-1, 1)).flatten()
    test_data_scaled = scaler.transform(test_data.reshape(-1, 1)).flatten()

    train_dataset = TimeSeriesDataset(train_data_scaled, window_size, horizon)
    val_dataset = TimeSeriesDataset(val_data_scaled, window_size, horizon)
    test_dataset = TimeSeriesDataset(test_data_scaled, window_size, horizon)

    return train_dataset, val_dataset, test_dataset, scaler


class MRNEnsemble:
    """
    Ensemble of MRN models for time series forecasting.

    Supports multiple ensemble strategies:
    - Simple averaging
    - Weighted averaging (based on validation performance)
    - Median ensemble
    - Best-of-N selection
    """

    def __init__(self, device: torch.device):
        self.models: List[MRN] = []
        self.model_configs: List[Dict] = []
        self.model_weights: List[float] = []
        self.device = device

    def add_model(self, model: MRN, config: Dict, weight: float = 1.0):
        """Add a model to the ensemble."""
        self.models.append(model)
        self.model_configs.append(config)
        self.model_weights.append(weight)

    def predict(
        self,
        inputs: torch.Tensor,
        method: str = "average",
        return_individual: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Make predictions using the ensemble.

        Args:
            inputs: Input tensor [batch, seq_len, features]
            method: Ensemble method ('average', 'weighted', 'median', 'best')
            return_individual: If True, also return individual model predictions

        Returns:
            Ensemble predictions (and optionally individual predictions)
        """
        all_predictions = []

        for model in self.models:
            model.eval()
            with torch.no_grad():
                model.reset_memory()
                pred = model(inputs, return_sequences=True)
                all_predictions.append(pred)

        # Stack predictions: [num_models, batch, seq_len, features]
        all_predictions = torch.stack(all_predictions)

        # Compute ensemble prediction
        if method == "average":
            ensemble_pred = all_predictions.mean(dim=0)

        elif method == "weighted":
            # Weighted average using model weights
            weights = torch.tensor(self.model_weights, device=self.device)
            weights = weights / weights.sum()  # Normalize
            weights = weights.view(-1, 1, 1, 1)  # Reshape for broadcasting
            ensemble_pred = (all_predictions * weights).sum(dim=0)

        elif method == "median":
            ensemble_pred = all_predictions.median(dim=0)[0]

        elif method == "best":
            # Use predictions from the best model (first in the list)
            ensemble_pred = all_predictions[0]

        else:
            raise ValueError(f"Unknown ensemble method: {method}")

        if return_individual:
            return ensemble_pred, all_predictions
        return ensemble_pred

    def compute_diversity(self, inputs: torch.Tensor) -> Dict[str, float]:
        """
        Compute diversity metrics for the ensemble.

        Higher diversity often leads to better ensemble performance.
        """
        all_predictions = []

        for model in self.models:
            model.eval()
            with torch.no_grad():
                model.reset_memory()
                pred = model(inputs, return_sequences=True)
                all_predictions.append(pred.cpu().numpy())

        all_predictions = np.array(
            all_predictions
        )  # [num_models, batch, seq_len, features]

        # Compute pairwise disagreement
        num_models = len(self.models)
        disagreements = []

        for i in range(num_models):
            for j in range(i + 1, num_models):
                diff = np.abs(all_predictions[i] - all_predictions[j])
                disagreements.append(diff.mean())

        avg_disagreement = np.mean(disagreements)

        # Compute coefficient of variation
        mean_pred = all_predictions.mean(axis=0)
        std_pred = all_predictions.std(axis=0)
        cv = np.mean(std_pred / (mean_pred + 1e-10))

        return {
            "avg_disagreement": avg_disagreement,
            "coefficient_of_variation": cv,
            "num_models": num_models,
        }

    def save(self, save_dir: Path):
        """Save all models in the ensemble."""
        save_dir.mkdir(parents=True, exist_ok=True)

        for i, (model, config) in enumerate(zip(self.models, self.model_configs)):
            model_path = save_dir / f"model_{i}.pth"
            torch.save(model.state_dict(), model_path)

            config_path = save_dir / f"model_{i}_config.json"
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)

        ensemble_info = {"num_models": len(self.models), "weights": self.model_weights}
        with open(save_dir / "ensemble_info.json", "w") as f:
            json.dump(ensemble_info, f, indent=2)

        print(f"Saved ensemble with {len(self.models)} models to {save_dir}")


def train_single_model(
    nn_structure: List[int],
    memory_structure: List[int],
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    model_name: str,
    verbose: bool = True,
) -> Tuple[MRN, float, List[float], List[float]]:
    """
    Train a single MRN model.

    Returns:
        Trained model, the best validation loss, train loss history, val loss history
    """
    model = MRN(
        nn_structure=nn_structure, memory_structure=memory_structure, device=device
    ).to(device)

    # Initialize weights
    for name, param in model.named_parameters():
        if "weight" in name:
            nn.init.xavier_uniform_(param)
        elif "bias" in name:
            nn.init.zeros_(param)
        elif "memory" in name:
            nn.init.uniform_(param, -0.1, 0.1)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    train_loss_history = []
    val_loss_history = []

    if verbose:
        print(f"\nTraining {model_name}:")
        print(f"  Architecture: {nn_structure}")
        print(f"  Memory: {memory_structure}")

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            model.reset_memory()

            outputs = model(batch_x, return_sequences=True)
            outputs = outputs.squeeze(-1)

            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_loss_history.append(train_loss)

        # Validate
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                model.reset_memory()
                outputs = model(batch_x, return_sequences=True)
                outputs = outputs.squeeze(-1)

                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_loss_history.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        if verbose and ((epoch + 1) % 20 == 0 or epoch == 0):
            print(
                f"  Epoch [{epoch + 1:3d}/{epochs}] | "
                f"Train: {train_loss:.6f} | Val: {val_loss:.6f}"
            )

    if verbose:
        print(f"  Best Val Loss: {best_val_loss:.6f}")

    return model, best_val_loss, train_loss_history, val_loss_history


def evaluate_predictions(
    predictions: np.ndarray, actuals: np.ndarray
) -> Dict[str, float]:
    """Compute evaluation metrics."""
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
    mae = np.mean(np.abs(predictions - actuals))
    mape = np.mean(np.abs((actuals - predictions) / (actuals + 1e-10))) * 100

    return {"rmse": float(rmse), "mae": float(mae), "mape": float(mape)}


def plot_ensemble_results(
    train_losses: List[List[float]],
    val_losses: List[List[float]],
    model_names: List[str],
    test_results: Dict,
    save_path: str,
):
    """Plot ensemble training and prediction results."""
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # Plot 1: Individual model training curves
    ax1 = fig.add_subplot(gs[0, :])
    for i, (name, train_loss, val_loss) in enumerate(
        zip(model_names, train_losses, val_losses)
    ):
        epochs = range(1, len(train_loss) + 1)
        ax1.plot(epochs, val_loss, label=f"{name} (Val)", linewidth=2, alpha=0.7)

    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Validation Loss (MSE)", fontsize=12)
    ax1.set_title("Individual Model Training Progress", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=9, ncol=3)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Test predictions - Individual models
    ax2 = fig.add_subplot(gs[1, 0])
    actuals = test_results["actuals"]
    time_steps = np.arange(len(actuals))

    ax2.plot(
        time_steps, actuals, label="Actual", linewidth=2.5, color="black", alpha=0.8
    )

    cmap = plt.get_cmap("tab10")
    colors = cmap(np.linspace(0, 1, len(model_names)))
    for i, name in enumerate(model_names):
        preds = test_results[f"{name}_predictions"]
        ax2.plot(
            time_steps, preds, label=name, linewidth=1.5, alpha=0.6, color=colors[i]
        )

    ax2.set_xlabel("Sequence Index", fontsize=11)
    ax2.set_ylabel("COVID-19 Cases", fontsize=11)
    ax2.set_title("Individual Model Predictions", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=8, loc="upper left")
    ax2.grid(True, alpha=0.3)

    # Plot 3: Test predictions - Ensemble methods
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(
        time_steps, actuals, label="Actual", linewidth=2.5, color="black", alpha=0.8
    )

    ensemble_methods = ["average", "weighted", "median"]
    ensemble_colors = ["red", "blue", "green"]

    for method, color in zip(ensemble_methods, ensemble_colors):
        preds = test_results[f"ensemble_{method}_predictions"]
        ax3.plot(
            time_steps,
            preds,
            label=f"Ensemble ({method})",
            linewidth=2,
            alpha=0.8,
            color=color,
            linestyle="--",
        )

    ax3.set_xlabel("Sequence Index", fontsize=11)
    ax3.set_ylabel("COVID-19 Cases", fontsize=11)
    ax3.set_title("Ensemble Predictions", fontsize=13, fontweight="bold")
    ax3.legend(fontsize=9, loc="upper left")
    ax3.grid(True, alpha=0.3)

    # Plot 4: Performance comparison
    ax4 = fig.add_subplot(gs[2, 0])

    all_names = model_names + [f"Ensemble ({m})" for m in ensemble_methods]
    rmse_values = []
    mae_values = []

    for name in model_names:
        metrics = test_results[f"{name}_metrics"]
        rmse_values.append(metrics["rmse"])
        mae_values.append(metrics["mae"])

    for method in ensemble_methods:
        metrics = test_results[f"ensemble_{method}_metrics"]
        rmse_values.append(metrics["rmse"])
        mae_values.append(metrics["mae"])

    x = np.arange(len(all_names))
    width = 0.35

    ax4.bar(x - width / 2, rmse_values, width, label="RMSE", alpha=0.8)
    ax4.bar(x + width / 2, mae_values, width, label="MAE", alpha=0.8)

    ax4.set_xlabel("Model", fontsize=11)
    ax4.set_ylabel("Error", fontsize=11)
    ax4.set_title("Performance Comparison", fontsize=13, fontweight="bold")
    ax4.set_xticks(x)
    ax4.set_xticklabels(all_names, rotation=45, ha="right", fontsize=8)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3, axis="y")

    # Plot 5: Prediction intervals (ensemble uncertainty)
    ax5 = fig.add_subplot(gs[2, 1])

    # Get all individual predictions to compute uncertainty
    individual_preds = []
    for name in model_names:
        individual_preds.append(test_results[f"{name}_predictions"])

    individual_preds = np.array(individual_preds)
    mean_pred = individual_preds.mean(axis=0)
    std_pred = individual_preds.std(axis=0)

    ax5.plot(time_steps, actuals, label="Actual", linewidth=2, color="black", alpha=0.8)
    ax5.plot(
        time_steps,
        mean_pred,
        label="Ensemble Mean",
        linewidth=2,
        color="red",
        linestyle="--",
    )
    ax5.fill_between(
        time_steps,
        mean_pred - 2 * std_pred,
        mean_pred + 2 * std_pred,
        alpha=0.3,
        color="red",
        label="±2σ Confidence",
    )

    ax5.set_xlabel("Sequence Index", fontsize=11)
    ax5.set_ylabel("COVID-19 Cases", fontsize=11)
    ax5.set_title(
        "Ensemble Uncertainty (Prediction Intervals)", fontsize=13, fontweight="bold"
    )
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)

    plt.suptitle(
        "MRN Ensemble Performance Analysis", fontsize=16, fontweight="bold", y=0.995
    )
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train MRN Ensemble for Time Series")

    # Data parameters
    parser.add_argument("--csv_path", type=str, default="../data/time_series/c19.csv")
    parser.add_argument("--country", type=str, default="Afghanistan")
    parser.add_argument("--window_size", type=int, default=14)
    parser.add_argument("--horizon", type=int, default=7)

    # Ensemble parameters
    parser.add_argument(
        "--num_models", type=int, default=5, help="Number of models in ensemble"
    )
    parser.add_argument(
        "--diverse_architectures",
        action="store_true",
        help="Use different architectures for each model",
    )

    # Training parameters
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)

    # Output
    parser.add_argument("--output_dir", type=str, default="../data/ensemble/")

    args = parser.parse_args()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    print("=" * 70)
    print("MRN Ensemble Training")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    case_data = load_data(args.csv_path, args.country)
    train_dataset, val_dataset, test_dataset, scaler = prepare_data(
        case_data, args.window_size, args.horizon
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Define model architectures
    if args.diverse_architectures:
        architectures = [
            ([1, 16, 8, 1], [3, 2, 1, 0]),
            ([1, 32, 16, 1], [4, 3, 2, 0]),
            ([1, 16, 32, 1], [2, 3, 2, 0]),
            ([1, 24, 12, 1], [3, 2, 1, 0]),
            ([1, 32, 8, 1], [4, 2, 1, 0]),
        ]
    else:
        # Same architecture, different random initializations
        base_arch = ([1, 16, 32, 1], [4, 3, 2, 0])
        architectures = [base_arch] * args.num_models

    # Train ensemble
    print("\n" + "=" * 70)
    print("Training Ensemble Models")
    print("=" * 70)

    ensemble = MRNEnsemble(device)
    all_train_losses = []
    all_val_losses = []
    model_names = []

    for i in range(args.num_models):
        nn_struct, mem_struct = architectures[i]
        model_name = f"Model-{i + 1}"
        model_names.append(model_name)

        # Train model
        model, val_loss, train_history, val_history = train_single_model(
            nn_struct,
            mem_struct,
            train_loader,
            val_loader,
            device,
            args.epochs,
            args.lr,
            model_name,
            verbose=True,
        )

        # Add to the ensemble (weight by inverse validation loss)
        weight = 1.0 / (val_loss + 1e-6)
        ensemble.add_model(
            model,
            {"nn_structure": nn_struct, "memory_structure": mem_struct},
            weight=weight,
        )

        # Track losses
        all_train_losses.append(train_history)
        all_val_losses.append(val_history)

    # Evaluate ensemble
    print("\n" + "=" * 70)
    print("Evaluating Ensemble")
    print("=" * 70)

    # Collect test predictions
    test_results: Dict[str, Any] = {"actuals": []}

    # Individual model predictions
    for i, (model, name) in enumerate(zip(ensemble.models, model_names)):
        model.eval()
        predictions = []
        actuals = []

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                model.reset_memory()
                outputs = model(batch_x, return_sequences=True)

                pred = outputs[:, -1, 0].cpu().numpy()
                actual = batch_y[:, -1].cpu().numpy()

                predictions.extend(pred)
                actuals.extend(actual)

        predictions = scaler.inverse_transform(
            np.array(predictions).reshape(-1, 1)
        ).flatten()
        actuals_orig = scaler.inverse_transform(
            np.array(actuals).reshape(-1, 1)
        ).flatten()

        if i == 0:
            test_results["actuals"] = actuals_orig

        test_results[f"{name}_predictions"] = predictions
        test_results[f"{name}_metrics"] = evaluate_predictions(
            predictions, actuals_orig
        )

    # Ensemble predictions
    ensemble_methods = ["average", "weighted", "median"]

    for method in ensemble_methods:
        predictions = []

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device)

                ensemble_output = ensemble.predict(batch_x, method=method)
                pred = ensemble_output[:, -1, 0].cpu().numpy()
                predictions.extend(pred)

        predictions = scaler.inverse_transform(
            np.array(predictions).reshape(-1, 1)
        ).flatten()

        test_results[f"ensemble_{method}_predictions"] = predictions
        test_results[f"ensemble_{method}_metrics"] = evaluate_predictions(
            predictions, test_results["actuals"]
        )

    # Print results
    print("\nTest Set Performance:")
    print("-" * 70)

    for name in model_names:
        metrics = test_results[f"{name}_metrics"]
        print(
            f"{name:12s} | RMSE: {metrics['rmse']:7.2f} | "
            f"MAE: {metrics['mae']:7.2f} | MAPE: {metrics['mape']:6.2f}%"
        )

    print("-" * 70)

    for method in ensemble_methods:
        metrics = test_results[f"ensemble_{method}_metrics"]
        print(
            f"Ensemble ({method:8s}) | RMSE: {metrics['rmse']:7.2f} | "
            f"MAE: {metrics['mae']:7.2f} | MAPE: {metrics['mape']:6.2f}%"
        )

    # Compute diversity
    print("\n" + "=" * 70)
    print("Ensemble Diversity Analysis")
    print("=" * 70)

    # Sample batch for diversity computation
    sample_batch = next(iter(test_loader))[0].to(device)
    diversity = ensemble.compute_diversity(sample_batch)

    print(f"Number of models: {diversity['num_models']}")
    print(f"Average disagreement: {diversity['avg_disagreement']:.6f}")
    print(f"Coefficient of variation: {diversity['coefficient_of_variation']:.6f}")

    # Plot results
    print("\n" + "=" * 70)
    print("Generating Plots")
    print("=" * 70)

    plot_ensemble_results(
        all_train_losses,
        all_val_losses,
        model_names,
        test_results,
        str(output_dir / "ensemble_results.png"),
    )

    # Save ensemble
    ensemble.save(output_dir / "models")

    print("\n" + "=" * 70)
    print("Ensemble Training Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
