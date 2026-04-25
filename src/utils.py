from typing import List, Optional, TYPE_CHECKING

import torch
import subprocess

if TYPE_CHECKING:
    from src.model.mrn import MRN


def print_gpu_memory():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            total_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            memory_used = torch.cuda.memory_allocated(i) / 1e9
            memory_reserved = torch.cuda.memory_reserved(i) / 1e9
            print(
                f"GPU {i}: Used {memory_used:.2f}GB | Reserved {memory_reserved:.2f}GB | Total {total_memory:.2f}GB"
            )

    elif torch.backends.mps.is_available():
        result = subprocess.run(
            ["sysctl", "hw.memsize"], capture_output=True, text=True
        )
        total_memory = int(result.stdout.split(":")[1].strip()) / 1e9
        memory_used = torch.mps.current_allocated_memory() / 1e9
        print(f"GPU 0: Used {memory_used:.2f}GB | Total {total_memory:.2f}GB")
    else:
        print("No GPU available - running on CPU")


def create_mrn_from_structure(
    nn_structure: List[int],
    mm_structure: List[int],
    device: Optional[torch.device] = None,
) -> "MRN":
    """Create an MRN using the structure notation."""
    from src.model.mrn import MRN

    if len(nn_structure) < 3:
        raise ValueError(
            f"nn_structure must have at least 3 elements, got {len(nn_structure)}"
        )

    return MRN(nn_structure=nn_structure, memory_structure=mm_structure, device=device)


def print_mrn_info(model: "MRN"):
    """Print information about an MRN model."""
    print(f"MRN Model Information:")
    print(f"  Network structure: {model.nn_structure}")
    print(f"  Memory structure: {model.cell.memory_structure}")
    print(f"  Number of layers: {model.num_layers}")
    print(f"  Input size: {model.input_size}")
    print(f"  Output size: {model.output_size}")
    print(f"  Device: {model.device}")

    print(f"\nLayer Details:")
    for i in range(model.num_layers):
        layer_type = (
            "Input"
            if i == 0
            else ("Output" if i == model.num_layers - 1 else f"Hidden {i}")
        )
        mem_info = (
            f"{model.cell.memory_structure[i]} memories"
            if model.cell.memory_structure[i] > 0
            else "no memory"
        )
        print(f"  Layer {i} ({layer_type}): size={model.nn_structure[i]}, {mem_info}")

    print(f"\nMemory Banks:")
    if len(model.cell.memory_banks) == 0:
        print("  No memory banks configured")
    else:
        for layer_idx, bank in sorted(
            model.cell.memory_banks.items(), key=lambda x: int(x[0])
        ):
            layer_num = int(layer_idx)
            layer_type = (
                "Input"
                if layer_num == 0
                else (
                    "Output"
                    if layer_num == model.num_layers - 1
                    else f"Hidden {layer_num}"
                )
            )
            targets = dict(bank.target_layer_sizes)
            print(
                f"  Layer {layer_idx} ({layer_type}): {bank.num_items} items × {bank.layer_size} dims → hidden layers {targets}"
            )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nParameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")