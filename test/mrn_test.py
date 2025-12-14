import torch

from src.model.mrn import MRN
from src.utils import print_mrn_info, create_mrn_from_structure

# Example usage
print("=" * 70)
print("MRN Cell-Based Implementation - Variable Layer Architecture")
print("=" * 70)

# Example 1: Basic 3-layer network (minimum)
print("\n" + "=" * 70)
print("Example 1: Basic 3-Layer Network")
print("=" * 70)

nn_structure_1 = [10, 20, 5]  # input=10, hidden=20, output=5
memory_structure_1 = [4, 3, 2]  # 4 input memories, 3 hidden memories, 2 output memories

model_1 = MRN(nn_structure=nn_structure_1, memory_structure=memory_structure_1)

print_mrn_info(model_1)

# Example 2: Deep network with 5 layers
print("\n" + "=" * 70)
print("Example 2: Deep 5-Layer Network")
print("=" * 70)

nn_structure_2 = [10, 20, 30, 15, 5]  # input + 3 hidden + output
memory_structure_2 = [4, 3, 2, 0, 1]  # memories at layers 0,1,2,4 but not 3

model_2 = MRN(nn_structure=nn_structure_2, memory_structure=memory_structure_2)

print_mrn_info(model_2)

# Example 3: Auto-padding memory structure
print("\n" + "=" * 70)
print("Example 3: Auto-Padding Memory Structure")
print("=" * 70)

nn_structure_3 = [10, 20, 30, 15, 5]  # 5 layers
memory_structure_3 = [4, 3]  # Only specify first 2, rest auto-padded with 0

model_3 = MRN(nn_structure=nn_structure_3, memory_structure=memory_structure_3)

print_mrn_info(model_3)
print(
    f"\nNote: memory_structure auto-padded from {memory_structure_3} to {model_3.cell.memory_structure}"
)

# Test with a single timestep
print("\n" + "=" * 70)
print("Single Timestep Test (using model 2)")
print("=" * 70)

single_input = torch.randn(model_2.input_size)
output, activations, state = model_2.cell(single_input)

print(f"Input shape: {single_input.shape}")
print(f"Output shape: {output.shape}")
print(f"\nActivations at each layer:")
for layer_idx, act in activations.items():
    layer_type = (
        "Input"
        if layer_idx == 0
        else (
            "Output" if layer_idx == model_2.num_layers - 1 else f"Hidden {layer_idx}"
        )
    )
    print(f"  Layer {layer_idx} ({layer_type}): {act.shape}")
print(f"\nState memory banks: {list(state.memory_banks.keys())}")

# Test with a sequence
print("\n" + "=" * 70)
print("Sequence Processing Test")
print("=" * 70)

seq_len = 15
batch_size = 3
sequence_input = torch.randn(batch_size, seq_len, model_2.input_size)

# Get outputs only
outputs = model_2(sequence_input)
print(f"Input shape: {sequence_input.shape}")
print(f"Outputs shape: {outputs.shape}")

# Get outputs and activations
outputs, all_activations = model_2(sequence_input, return_activations=True)
print(f"\nWith activations:")
print(f"  Outputs shape: {outputs.shape}")
for layer_idx, acts in all_activations.items():
    layer_type = (
        "Input"
        if layer_idx == 0
        else (
            "Output" if layer_idx == model_2.num_layers - 1 else f"Hidden {layer_idx}"
        )
    )
    print(f"  Layer {layer_idx} ({layer_type}) activations: {acts.shape}")

# Get outputs and state
outputs, final_state = model_2(sequence_input, return_state=True)
print(f"\nWith final state:")
print(f"  Outputs shape: {outputs.shape}")
print(f"  Final state memory banks: {list(final_state.memory_banks.keys())}")

# Test return last only
last_output = model_2(sequence_input, return_sequences=False)
print(f"\nLast output only: {last_output.shape}")

# Example 4: Using create_mrn_from_structure helper
print("\n" + "=" * 70)
print("Example 4: Using Helper Function")
print("=" * 70)

model_4 = create_mrn_from_structure(
    nn_structure=[8, 16, 32, 16, 4],
    mm_structure=[2, 3, 1],  # Will be padded to [2, 3, 1, 0, 0]
)

print_mrn_info(model_4)

print("\n" + "=" * 70)
print("All examples completed successfully!")
print("=" * 70)
