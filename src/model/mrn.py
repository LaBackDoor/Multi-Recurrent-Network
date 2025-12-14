from typing import List, Optional, Tuple

import torch
from torch import nn

from src.model.cell import MRNCell, MRNState


class MRN(nn.Module):
    """
    Multi-Recurrent Neural Network - processes sequences.

    This wraps MRNCell to process entire sequences

    Args:
        nn_structure: List of layer sizes [input_size, hidden1, ..., output_size]
                     Minimum length is 3
        memory_structure: List of memory bank sizes for each layer
                         Auto-padded with zeros if shorter than nn_structure
        device: Device to create tensors on
    """

    def __init__(
        self,
        nn_structure: List[int],
        memory_structure: List[int],
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        self.nn_structure = nn_structure
        self.memory_structure = memory_structure
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Convenience properties
        self.input_size = nn_structure[0]
        self.output_size = nn_structure[-1]
        self.num_layers = len(nn_structure)

        # Create the cell
        self.cell = MRNCell(
            nn_structure=nn_structure,
            memory_structure=memory_structure,
            device=self.device,
        )

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[MRNState] = None,
        return_sequences: bool = True,
        return_state: bool = False,
        return_activations: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Process a sequence.

        Args:
            inputs: Input sequence [batch_size, seq_len, input_size] or [seq_len, input_size]
            states: Initial state (if None, uses current memory state)
            return_sequences: If True, return all outputs; if False, return only last
            return_state: If True, also return final state
            return_activations: If True, also return all layer activations

        Returns:
            output: Output sequence or final output
            activations: (optional) Dictionary of layer activations if return_activations=True
            state: (optional) Final state if return_state=True
        """
        # Handle input dimensions
        if inputs.dim() == 2:
            # [seq_len, input_size] -> [1, seq_len, input_size]
            inputs = inputs.unsqueeze(0)
            squeeze_batch = True
        else:
            squeeze_batch = False

        _, sequence_length, _ = inputs.shape

        # Initialize state if not provided
        if states is None:
            states = self.cell.init_state()

        # Process sequence
        outputs_values = []
        layer_activations_list = {i: [] for i in range(self.num_layers)}

        for t in range(sequence_length):
            cell_output, cell_activations, states = self.cell(inputs[:, t], states)
            outputs_values.append(cell_output)

            # Store activations for each layer
            for layer_index, activation in cell_activations.items():
                layer_activations_list[layer_index].append(activation)

        # Stack outputs
        outputs_values = torch.stack(
            outputs_values, dim=1
        )  # [batch_size, seq_len, output_size]

        # Stack all activations into a new dict
        all_layer_activations = {
            layer_index: torch.stack(activations, dim=1)
            for layer_index, activations in layer_activations_list.items()
        }

        if squeeze_batch:
            outputs_values = outputs_values.squeeze(0)
            all_layer_activations = {
                k: v.squeeze(0) for k, v in all_layer_activations.items()
            }

        # Prepare return values
        if not return_sequences:
            outputs_values = (
                outputs_values[:, -1] if not squeeze_batch else outputs_values[-1]
            )
            all_layer_activations = {
                k: (v[:, -1] if not squeeze_batch else v[-1])
                for k, v in all_layer_activations.items()
            }

        # Build return tuple
        result = (outputs_values,)

        if return_activations:
            result = result + (all_layer_activations,)

        if return_state:
            result = result + (states,)

        return result if len(result) > 1 else result[0]

    def reset_memory(self):
        """Reset all memory banks to random values."""
        with torch.no_grad():
            for bank in self.cell.memory_banks.values():
                # Create a new random tensor and copy to avoid in-place operation issues
                new_memory = torch.rand_like(bank.memory)
                bank.memory.copy_(new_memory)

    def set_update_memory(self, update: bool):
        """
        Enable or disable memory updates during forward passes.

        This is useful during autoregressive generation where we want to
        accumulate outputs without updating memory between steps.

        Args:
            update: If True, memory will be updated; if False, memory stays frozen
        """
        self.cell._update_memory_flag = update
