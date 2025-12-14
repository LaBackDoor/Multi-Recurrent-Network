"""
Train Multi-Recurrent Neural Network (MRN) for Character-Level Text Generation

This task USES softmax for:
1. Training: Computing probability distributions over characters
2. Generation: Sampling from softmax distributions
3. Temperature scaling: Controlling randomness in generation
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.model.mrn import MRN


class CharDataset(Dataset):
    """
    Character-level dataset for text generation.

    Creates sequences where:
    - Input: sequence of characters [0, seq_len-1]
    - Target: next characters [1, seq_len] (shifted by 1)

    Args:
        text: Raw text string
        seq_length: Length of input sequences
        char_to_idx: Character to index mapping
    """

    def __init__(self, text: str, seq_length: int, char_to_idx: Dict[str, int]):
        self.text = text
        self.seq_length = seq_length
        self.char_to_idx = char_to_idx
        self.sequences = self._create_sequences()

    def _create_sequences(self) -> List[Tuple[List[int], List[int]]]:
        """Create overlapping sequences."""
        sequences = []

        for i in range(len(self.text) - self.seq_length):
            # Input: characters [i, i+seq_length]
            input_seq = [
                self.char_to_idx.get(c, 0) for c in self.text[i : i + self.seq_length]
            ]
            # Target: characters [i+1, i+seq_length+1] (predict the next char at each step)
            target_seq = [
                self.char_to_idx.get(c, 0)
                for c in self.text[i + 1 : i + self.seq_length + 1]
            ]
            sequences.append((input_seq, target_seq))

        return sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        input_seq, target_seq = self.sequences[idx]
        return (
            torch.LongTensor(input_seq),  # [seq_length]
            torch.LongTensor(target_seq),  # [seq_length]
        )


def prepare_data(
    text_path: str, seq_length: int, val_split: float = 0.1
) -> Tuple[CharDataset, CharDataset, Dict[str, int], Dict[int, str], int]:
    """
    Load and prepare text data.

    Returns:
        train_dataset, val_dataset, char_to_idx, idx_to_char, vocab_size
    """
    # Load text
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"Loaded text: {len(text):,} characters")

    # Create vocabulary
    unique_chars = sorted(set(text))
    vocab_size = len(unique_chars)

    char_to_idx = {ch: idx for idx, ch in enumerate(unique_chars)}
    idx_to_char = {idx: ch for ch, idx in char_to_idx.items()}

    print(f"Vocabulary size: {vocab_size}")
    print(f"Characters: {repr(''.join(unique_chars[:50]))}")
    if vocab_size > 50:
        print(f"           ... and {vocab_size - 50} more")

    # Split into train/val
    split_idx = int(len(text) * (1 - val_split))
    train_text = text[:split_idx]
    val_text = text[split_idx:]

    print(f"\nData split:")
    print(f"  Train: {len(train_text):,} characters")
    print(f"  Val:   {len(val_text):,} characters")

    # Create datasets
    train_dataset = CharDataset(train_text, seq_length, char_to_idx)
    val_dataset = CharDataset(val_text, seq_length, char_to_idx)

    print(f"\nSequences created:")
    print(f"  Train: {len(train_dataset):,} sequences")
    print(f"  Val:   {len(val_dataset):,} sequences")

    return train_dataset, val_dataset, char_to_idx, idx_to_char, vocab_size


def sample_from_logits(logits: torch.Tensor, temperature: float = 1.0) -> int:
    """
    Sample a character index from logits using softmax with temperature.

    This is where SOFTMAX is used for generation!

    Args:
        logits: Raw output scores [vocab_size]
        temperature: Controls randomness
            - Low (0.5): More deterministic, conservative
            - 1.0: Standard sampling
            - High (1.5+): More random, creative

    Returns:
        Sampled character index
    """
    # Apply temperature scaling
    logits = logits / temperature

    # Apply softmax to get probability distribution
    probabilities = torch.softmax(logits, dim=-1)

    # Sample from the distribution
    sampled_idx = torch.multinomial(probabilities, num_samples=1).item()

    return sampled_idx


def generate_text(
    model: MRN,
    seed_text: str,
    char_to_idx: Dict[str, int],
    idx_to_char: Dict[int, str],
    device: torch.device,
    length: int = 200,
    temperature: float = 1.0,
) -> str:
    """
    Generate text using the trained model.

    Uses softmax + temperature sampling at each step.

    Args:
        model: Trained MRN model
        seed_text: Starting text
        char_to_idx: Character to index mapping
        idx_to_char: Index to character mapping
        device: Device to run on
        length: Number of characters to generate
        temperature: Sampling temperature

    Returns:
        Generated text string
    """
    model.eval()

    # Convert seed to indices
    current_seq = [char_to_idx.get(c, 0) for c in seed_text]
    generated = seed_text

    # Reset model memory
    model.reset_memory()

    with torch.no_grad():
        for _ in range(length):
            # Prepare input (use last seq_length characters)
            seq_len = model.input_size
            input_seq = (
                current_seq[-seq_len:] if len(current_seq) >= seq_len else current_seq
            )

            # Pad if needed
            while len(input_seq) < seq_len:
                input_seq.insert(0, 0)

            # Create one-hot encoding
            input_tensor = torch.zeros(
                1, int(seq_len), int(model.input_size), device=device
            )
            for i, idx in enumerate(input_seq):
                if idx < model.input_size:
                    input_tensor[0, i, idx] = 1.0

            # Get model output (last timestep only)
            output = model(input_tensor, return_sequences=False)  # [1, vocab_size]

            # Sample using softmax plus temperature
            next_idx = sample_from_logits(output[0], temperature=temperature)

            # Append to sequence and generated text
            current_seq.append(next_idx)
            generated += idx_to_char.get(next_idx, "")

    return generated


def train_epoch(
    model: MRN,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    vocab_size: int,
    gradient_norm: float = 5.0,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    for batch_input, batch_target in data_loader:
        batch_input = batch_input.to(device)  # [batch, seq_len]
        batch_target = batch_target.to(device)  # [batch, seq_len]

        batch_size, seq_len = batch_input.shape

        # Convert to one-hot encoding [batch, seq_len, vocab_size]
        input_one_hot = torch.zeros(batch_size, seq_len, vocab_size, device=device)
        input_one_hot.scatter_(2, batch_input.unsqueeze(-1), 1.0)

        optimizer.zero_grad()

        # Reset memory for each sequence
        model.reset_memory()

        # Get predictions for all timesteps
        outputs = model(
            input_one_hot, return_sequences=True
        )  # [batch, seq_len, vocab_size]

        # Reshape for loss computation
        outputs = outputs.view(-1, vocab_size)  # [batch*seq_len, vocab_size]
        targets = batch_target.view(-1)  # [batch*seq_len]

        # CrossEntropyLoss applies softmax internally
        loss = criterion(outputs, targets)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_norm)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(data_loader)


def evaluate(
    model: MRN,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    vocab_size: int,
) -> float:
    """Evaluate model."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch_input, batch_target in data_loader:
            batch_input = batch_input.to(device)
            batch_target = batch_target.to(device)

            batch_size, seq_len = batch_input.shape

            # One-hot encoding
            input_one_hot = torch.zeros(batch_size, seq_len, vocab_size, device=device)
            input_one_hot.scatter_(2, batch_input.unsqueeze(-1), 1.0)

            # Reset memory
            model.reset_memory()

            # Get predictions
            outputs = model(input_one_hot, return_sequences=True)

            # Compute loss
            outputs = outputs.view(-1, vocab_size)
            targets = batch_target.view(-1)
            loss = criterion(outputs, targets)

            total_loss += loss.item()

    return total_loss / len(data_loader)


def create_sample_text(output_dir: Path) -> str:
    """Create a sample text file if none exists."""
    sample_path = output_dir / "sample_text.txt"

    if not sample_path.exists():
        # Create sample text (Shakespeare-style)
        sample_text = (
            """
        To be, or not to be, that is the question:
        Whether 'tis nobler in the mind to suffer
        The slings and arrows of outrageous fortune,
        Or to take arms against a sea of troubles
        And by opposing end them. To die—to sleep,
        No more; and by a sleep to say we end
        The heart-ache and the thousand natural shocks
        That flesh is heir to: 'tis a consummation
        Devoutly to be wished. To die, to sleep;
        To sleep, perchance to dream—ay, there's the rub:
        For in that sleep of death what dreams may come,
        When we have shuffled off this mortal coil,
        Must give us pause—there's the respect
        That makes calamity of so long life.
        """
            * 10
        )  # Repeat to have more data

        sample_path.write_text(sample_text)
        print(f"Created sample text file: {sample_path}")

    return str(sample_path)


def main():
    parser = argparse.ArgumentParser(description="Train MRN for Text Generation")

    # Data parameters
    parser.add_argument(
        "--text_path",
        type=str,
        default=None,
        help="Path to text file (creates sample if not provided)",
    )
    parser.add_argument(
        "--seq_length", type=int, default=50, help="Length of input sequences"
    )
    parser.add_argument(
        "--val_split", type=float, default=0.1, help="Fraction for validation"
    )

    # Model parameters
    parser.add_argument(
        "--hidden_sizes",
        type=int,
        nargs="+",
        default=[128, 64],
        help="Sizes of hidden layers",
    )
    parser.add_argument(
        "--memory_structure",
        type=int,
        nargs="+",
        default=[4, 3, 2, 0],
        help="Memory banks per layer",
    )

    # Training parameters
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.002, help="Learning rate")
    parser.add_argument(
        "--gradient_norm", type=float, default=5.0, help="Gradient clipping"
    )

    # Generation parameters
    parser.add_argument(
        "--gen_length", type=int, default=300, help="Length of generated text"
    )
    parser.add_argument(
        "--temperatures",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 1.5],
        help="Temperatures for generation",
    )
    parser.add_argument(
        "--seed_text",
        type=str,
        default="To be or not to be",
        help="Seed text for generation",
    )

    # Output
    parser.add_argument(
        "--output_dir", type=str, default="../data/text_gen/", help="Output directory"
    )

    args = parser.parse_args()

    # Create an output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print("\n" + "=" * 70)
    print("Preparing Data")
    print("=" * 70)

    if args.text_path is None:
        args.text_path = create_sample_text(output_dir)

    train_dataset, val_dataset, char_to_idx, idx_to_char, vocab_size = prepare_data(
        args.text_path, args.seq_length, args.val_split
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Create model
    print("\n" + "=" * 70)
    print("Creating MRN Model")
    print("=" * 70)

    # Network: [vocab_size, hidden..., vocab_size]
    nn_structure = [vocab_size] + args.hidden_sizes + [vocab_size]

    model = MRN(
        nn_structure=nn_structure, memory_structure=args.memory_structure, device=device
    ).to(device)

    # Initialize weights
    for name, param in model.named_parameters():
        if "weight" in name:
            nn.init.xavier_uniform_(param)
        elif "bias" in name:
            nn.init.zeros_(param)
        elif "memory" in name:
            nn.init.uniform_(param, -0.1, 0.1)

    print(f"Network structure: {nn_structure}")
    print(f"Memory structure: {model.cell.memory_structure}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()  # Uses softmax internally
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    print("\n" + "=" * 70)
    print("Training")
    print("=" * 70)

    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        train_loss = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            vocab_size,
            args.gradient_norm,
        )
        val_loss = evaluate(model, val_loader, criterion, device, vocab_size)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_dir / "best_text_gen_model.pth")

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch [{epoch + 1:3d}/{args.epochs}] | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
            )

    # Load the best model
    model.load_state_dict(torch.load(output_dir / "best_text_gen_model.pth"))

    # Generate text with different temperatures
    print("\n" + "=" * 70)
    print("Text Generation with Different Temperatures")
    print("=" * 70)
    print(f"\nSeed text: '{args.seed_text}'")
    print("=" * 70)

    for temp in args.temperatures:
        print(f"\n{'=' * 70}")
        print(
            f"Temperature: {temp} {'(deterministic)' if temp < 0.7 else '(creative)' if temp > 1.2 else '(balanced)'}"
        )
        print(f"{'=' * 70}")

        generated = generate_text(
            model,
            args.seed_text,
            char_to_idx,
            idx_to_char,
            device,
            length=args.gen_length,
            temperature=temp,
        )

        print(generated)
        print()

        # Save to file
        output_file = output_dir / f"generated_temp_{temp}.txt"
        output_file.write_text(generated)
        print(f"Saved to: {output_file}")

    print("\n" + "=" * 70)
    print("Softmax Demonstration")
    print("=" * 70)

    # Show how softmax transforms logits
    print("\nExample: How softmax converts logits to probabilities")
    print("-" * 70)

    model.eval()
    with torch.no_grad():
        # Get sample input
        sample_input = [
            char_to_idx.get(c, 0) for c in args.seed_text[: args.seq_length]
        ]
        while len(sample_input) < args.seq_length:
            sample_input.insert(0, 0)

        input_tensor = torch.zeros(1, args.seq_length, vocab_size, device=device)
        for i, idx in enumerate(sample_input):
            if idx < vocab_size:
                input_tensor[0, i, idx] = 1.0

        model.reset_memory()
        logits = model(input_tensor, return_sequences=False)[0]  # [vocab_size]

        # Show the top 5 logits vs. probabilities
        top_logits, top_indices = torch.topk(logits, 5)
        probs = torch.softmax(logits, dim=-1)
        top_probs = probs[top_indices]

        print(f"Top 5 characters (by raw logits):")
        for i, (idx, logit, prob) in enumerate(zip(top_indices, top_logits, top_probs)):
            char = idx_to_char.get(idx.item(), "?")
            print(
                f"  {i + 1}. '{char}': logit={logit:.3f} → prob={prob:.4f} ({prob * 100:.2f}%)"
            )

        print(f"\nSum of all probabilities: {probs.sum():.6f} (should be 1.0)")
        print(f"Entropy: {-(probs * torch.log(probs + 1e-10)).sum():.4f}")

    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
