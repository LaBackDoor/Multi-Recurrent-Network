"""
Train Multi-Recurrent Neural Network (MRN) on Digit Classification
Uses sklearn's digits dataset (8x8 images of handwritten digits)
"""

import argparse
from pathlib import Path
from typing import Tuple, Union, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from torch.utils.data import Dataset, DataLoader
import seaborn as sns

from src.model.mrn import MRN


class DigitsDataset(Dataset):
    """
    Dataset wrapper for digit classification.

    Each 8x8 digit image is treated as a sequence of 8 timesteps,
    where each timestep contains 8 pixel values (one row).

    Args:
        images: Digit images [N, 64] flattened
        labels: Class labels [N]
        scaler: Optional scaler for normalization
        fit_scaler: Whether to fit the scaler on this data
    """

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        scaler: StandardScaler = None,
        fit_scaler: bool = False,
    ):
        # Reshape images to [N, 8, 8] for sequence processing
        self.images = images.reshape(-1, 8, 8)
        self.labels = labels

        # Normalize data
        if scaler is None:
            scaler = StandardScaler()
            fit_scaler = True

        if fit_scaler:
            # Fit scaler on flattened data
            scaler.fit(images)

        # Transform each image
        self.scaler = scaler
        normalized = scaler.transform(images)
        self.images = normalized.reshape(-1, 8, 8)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Return [8, 8] sequence and scalar label
        return (
            torch.FloatTensor(self.images[idx]),  # [seq_len=8, features=8]
            torch.LongTensor([self.labels[idx]])[0],  # Scalar class label
        )


def load_and_prepare_data(
    test_size: float = 0.2, val_size: float = 0.2
) -> Tuple[DigitsDataset, DigitsDataset, DigitsDataset, StandardScaler]:
    """
    Load digit dataset and split into train/val/test.

    Args:
        test_size: Fraction of data for test set
        val_size: Fraction of remaining data for validation

    Returns:
        train_dataset, val_dataset, test_dataset, scaler
    """
    # Load digits dataset
    digits = load_digits()
    x, y = digits.data, digits.target

    print(f"Loaded digits dataset:")
    print(f"  Total samples: {len(x)}")
    print(f"  Image shape: 8x8 pixels")
    print(f"  Classes: {len(np.unique(y))} (digits 0-9)")
    print(f"  Features per image: {x.shape[1]}")

    # Shuffle data
    indices = np.random.permutation(len(x))
    x, y = x[indices], y[indices]

    # Split data
    test_split = int(len(x) * (1 - test_size))
    x_train_val, y_train_val = x[:test_split], y[:test_split]
    x_test, y_test = x[test_split:], y[test_split:]

    val_split = int(len(x_train_val) * (1 - val_size))
    x_train, y_train = x_train_val[:val_split], y_train_val[:val_split]
    x_val, y_val = x_train_val[val_split:], y_train_val[val_split:]

    print(f"\nData split:")
    print(f"  Train: {len(x_train)} samples")
    print(f"  Val:   {len(x_val)} samples")
    print(f"  Test:  {len(x_test)} samples")

    # Create datasets with normalization
    scaler = StandardScaler()
    train_dataset = DigitsDataset(x_train, y_train, scaler=scaler, fit_scaler=True)
    val_dataset = DigitsDataset(x_val, y_val, scaler=scaler, fit_scaler=False)
    test_dataset = DigitsDataset(x_test, y_test, scaler=scaler, fit_scaler=False)

    print(f"\nNormalization (StandardScaler):")
    print(f"  Mean: {scaler.mean_[:5]}... (showing first 5)")
    print(f"  Std:  {scaler.scale_[:5]}... (showing first 5)")

    return train_dataset, val_dataset, test_dataset, scaler


def train_epoch(
    model: MRN,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_norm: float = 1.0,
    return_grad_stats: bool = False,
) -> Union[Tuple[float, float], Tuple[float, float, Dict[str, float]]]:
    """
    Train for one epoch.

    Returns:
        avg_loss, accuracy, (optional) grad_stats
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    grad_norms = []

    for batch_x, batch_y in data_loader:
        batch_x = batch_x.to(device)  # [batch_size, 8, 8]
        batch_y = batch_y.to(device)  # [batch_size]

        optimizer.zero_grad()

        # Reset memory at the start of each sequence
        model.reset_memory()

        # Process sequence - we only want the FINAL output for classification
        # return_sequences=False gives us just the last timestep output
        outputs = model(batch_x, return_sequences=False)  # [batch_size, num_classes]

        # Compute loss (CrossEntropyLoss expects raw logits)
        loss = criterion(outputs, batch_y)
        loss.backward()

        # Track gradient norm before clipping
        if return_grad_stats:
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm**0.5
            grad_norms.append(total_norm)

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_norm)

        optimizer.step()

        total_loss += loss.item()

        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        total += batch_y.size(0)
        correct += predicted.eq(batch_y).sum().item()

    avg_loss = total_loss / len(data_loader)
    accuracy = 100.0 * correct / total

    if return_grad_stats:
        grad_stats = {
            "mean": np.mean(grad_norms),
            "max": np.max(grad_norms),
            "min": np.min(grad_norms),
        }
        return avg_loss, accuracy, grad_stats

    return avg_loss, accuracy


def evaluate(
    model: MRN, data_loader: DataLoader, criterion: nn.Module, device: torch.device
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Evaluate model on validation/test set.

    Returns:
        avg_loss, accuracy, all_predictions, all_labels
    """
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # Reset memory
            model.reset_memory()

            # Get final output
            outputs = model(
                batch_x, return_sequences=False
            )  # [batch_size, num_classes]

            # Compute loss
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()

            # Get predictions
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_predictions) * 100.0

    return avg_loss, accuracy, np.array(all_predictions), np.array(all_labels)


def plot_results(
    train_losses: list,
    val_losses: list,
    train_accs: list,
    val_accs: list,
    test_predictions: np.ndarray,
    test_labels: np.ndarray,
    test_dataset: DigitsDataset,
    save_path: str,
):
    """
    Plot training curves, confusion matrix, and sample predictions.
    """
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Plot 1: Training and validation loss
    ax1 = fig.add_subplot(gs[0, :2])
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, label="Train Loss", linewidth=2, color="blue")
    ax1.plot(epochs, val_losses, label="Validation Loss", linewidth=2, color="orange")
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss (CrossEntropy)", fontsize=12)
    ax1.set_title("MRN Training Progress - Loss", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Training and validation accuracy
    ax2 = fig.add_subplot(gs[1, :2])
    ax2.plot(epochs, train_accs, label="Train Accuracy", linewidth=2, color="blue")
    ax2.plot(epochs, val_accs, label="Validation Accuracy", linewidth=2, color="orange")
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Accuracy (%)", fontsize=12)
    ax2.set_title("MRN Training Progress - Accuracy", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Confusion Matrix
    ax3 = fig.add_subplot(gs[2, :2])
    cm = confusion_matrix(test_labels, test_predictions)
    class_labels = [str(i) for i in range(10)]
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax3,
        xticklabels=class_labels,
        yticklabels=class_labels,
    )
    ax3.set_xlabel("Predicted Label", fontsize=12)
    ax3.set_ylabel("True Label", fontsize=12)
    ax3.set_title("Confusion Matrix (Test Set)", fontsize=14, fontweight="bold")

    # Plot 4-12: Sample predictions (show 9 examples)
    sample_indices = np.random.choice(len(test_dataset), 9, replace=False)

    for idx, sample_idx in enumerate(sample_indices):
        row = idx // 3
        ax = fig.add_subplot(gs[row, 2])

        # Get image and prediction
        image = test_dataset.images[sample_idx]  # [8, 8]
        true_label = test_labels[sample_idx]
        pred_label = test_predictions[sample_idx]

        # Plot image
        ax.imshow(image, cmap="gray")
        ax.axis("off")

        # Color code: green if correct, red if wrong
        color = "green" if true_label == pred_label else "red"
        title = f"True: {true_label}, Pred: {pred_label}"
        ax.set_title(title, fontsize=10, color=color, fontweight="bold")

    plt.suptitle(
        "MRN Digit Classification Results", fontsize=16, fontweight="bold", y=0.995
    )
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train MRN on Digit Classification")

    # Data parameters
    parser.add_argument(
        "--test_size", type=float, default=0.2, help="Fraction of data for test set"
    )
    parser.add_argument(
        "--val_size",
        type=float,
        default=0.2,
        help="Fraction of remaining data for validation",
    )

    # Model parameters
    parser.add_argument(
        "--hidden_sizes",
        type=int,
        nargs="+",
        default=[32, 16],
        help="Sizes of hidden layers",
    )
    parser.add_argument(
        "--memory_structure",
        type=int,
        nargs="+",
        default=[3, 2, 1, 0],
        help="Number of memory banks per layer (input, hidden1, hidden2, output)",
    )

    # Training parameters
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--weight_decay", type=float, default=1e-5, help="L2 regularization"
    )
    parser.add_argument(
        "--gradient_norm",
        type=float,
        default=5.0,
        help="Max gradient norm for clipping",
    )
    parser.add_argument(
        "--disable_memory",
        action="store_true",
        help="Train without memory updates (freeze memory)",
    )
    parser.add_argument(
        "--lr_warmup",
        type=int,
        default=10,
        help="Number of epochs for learning rate warmup",
    )

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../data/classification/",
        help="Directory to save results",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create an output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print("=" * 70)
    print("Loading Digits Dataset")
    print("=" * 70)
    train_dataset, val_dataset, test_dataset, scaler = load_and_prepare_data(
        test_size=args.test_size, val_size=args.val_size
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Create an MRN model
    print("\n" + "=" * 70)
    print("Creating MRN Model")
    print("=" * 70)

    # Network structure: [input_size, hidden..., output_size]
    # Input: 8 features (one row of pixels)
    # Output: 10 classes (digits 0-9)
    nn_structure = [8] + args.hidden_sizes + [10]

    model = MRN(
        nn_structure=nn_structure, memory_structure=args.memory_structure, device=device
    ).to(device)

    # Initialize weights with Xavier initialization
    def init_weights(m):
        if isinstance(m, nn.Parameter):
            if m.dim() >= 2:
                nn.init.xavier_uniform_(m)
            else:
                nn.init.zeros_(m)

    for name, param in model.named_parameters():
        if "weight" in name:
            nn.init.xavier_uniform_(param)
        elif "bias" in name:
            nn.init.zeros_(param)
        elif "memory" in name:
            # Initialize memory banks with small values
            nn.init.uniform_(param, -0.1, 0.1)

    init_weights(model)

    print(f"Network structure: {nn_structure}")
    print(f"Memory structure: {model.cell.memory_structure}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optionally disable memory
    if args.disable_memory:
        print("\n⚠️  Memory updates DISABLED - running without recurrent memory")
        model.set_update_memory(False)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()  # Includes LogSoftmax internally
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # Learning rate scheduler
    main_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6
    )

    # Warmup scheduler wrapper
    class WarmupScheduler:
        def __init__(self, ws_optimizer, warmup_epochs, base_lr):
            self.optimizer = ws_optimizer
            self.warmup_epochs = warmup_epochs
            self.base_lr = base_lr
            self.current_epoch = 0

        def step(self):
            self.current_epoch += 1
            if self.current_epoch <= self.warmup_epochs:
                lr = self.base_lr * (self.current_epoch / self.warmup_epochs)
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr

    warmup_scheduler = WarmupScheduler(optimizer, args.lr_warmup, args.lr)

    # Training loop
    print("\n" + "=" * 70)
    print("Training")
    print("=" * 70)

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val_acc = 0.0
    grad_stats = {}

    for epoch in range(args.epochs):
        # Warmup learning rate
        if epoch < args.lr_warmup:
            warmup_scheduler.step()

        # Check gradient stats every 10 epochs
        return_grads = epoch % 10 == 0

        if return_grads:
            train_loss, train_acc, grad_stats = train_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                args.gradient_norm,
                return_grad_stats=True,
            )
        else:
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device, args.gradient_norm
            )

        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Step scheduler (only after warmup)
        if epoch >= args.lr_warmup:
            main_scheduler.step(val_loss)

        # Save the best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_dir / "best_mrn_classifier.pth")

        if (epoch + 1) % 10 == 0 or epoch == 0:
            log_msg = (
                f"Epoch [{epoch+1:3d}/{args.epochs}] | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}"
            )

            if return_grads:
                log_msg += (
                    f" | Grad: {grad_stats['mean']:.3f} (max: {grad_stats['max']:.3f})"
                )

            print(log_msg)

    # Load the best model for final evaluation
    model.load_state_dict(torch.load(output_dir / "best_mrn_classifier.pth"))

    # Final test evaluation
    print("\n" + "=" * 70)
    print("Final Evaluation on Test Set")
    print("=" * 70)

    test_loss, test_acc, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device
    )

    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")

    # Print classification report
    print("\nClassification Report:")
    print(
        classification_report(
            test_labels,
            test_preds,
            target_names=[str(i) for i in range(10)],
            zero_division=0,  # Suppress warnings for classes with no predictions
        )
    )

    # Plot results
    print("\n" + "=" * 70)
    print("Generating Plots")
    print("=" * 70)

    plot_results(
        train_losses,
        val_losses,
        train_accs,
        val_accs,
        test_preds,
        test_labels,
        test_dataset,
        str(output_dir / "mrn_classification_results.png"),
    )

    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Final Test Accuracy: {test_acc:.2f}%")


if __name__ == "__main__":
    main()
