"""Multi-Recurrent Neural Network (MRN) - sequence wrapper around MRNCell."""

from typing import List, Optional, Tuple, Union

import torch
from torch import nn

from src.model.cell import MRNCell, MRNState
from src.model.device import default_device


class MRN(nn.Module):
    """
    Multi-Recurrent Neural Network.

    Wraps MRNCell to process whole sequences with BPTT.

    Args:
        nn_structure: layer sizes [input, hidden..., output]. Minimum 3.
        memory_structure: memories per layer (auto-padded with zeros).
        weight_init_range: half-width of the symmetric uniform init range
            for all feedforward and memory-projection weights. Default 0.01,
            matching the NumPy reference and the thesis ("very small").
        hidden_bias_init_value: constant value for the first hidden layer's
            bias. Default 0.5, matching NumPy. Pass None for small uniform.
        init_memory_mode: "random" (thesis text) or "constant" (NumPy).
        init_memory_value: value used when init_memory_mode == "constant".
        device: target device.
    """

    def __init__(
        self,
        nn_structure: List[int],
        memory_structure: List[int],
        weight_init_range: float = 0.01,
        hidden_bias_init_value: Optional[float] = 0.5,
        init_memory_mode: str = "random",
        init_memory_value: float = 0.5,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        self.nn_structure = nn_structure
        self.memory_structure = memory_structure
        self.device = device or default_device()

        self.input_size = nn_structure[0]
        self.output_size = nn_structure[-1]
        self.num_layers = len(nn_structure)

        self.cell = MRNCell(
            nn_structure=nn_structure,
            memory_structure=memory_structure,
            weight_init_range=weight_init_range,
            hidden_bias_init_value=hidden_bias_init_value,
            init_memory_mode=init_memory_mode,
            init_memory_value=init_memory_value,
            device=self.device,
        )

    def init_state(self, batch_size: int = 1) -> MRNState:
        """Create a fresh initial state for a batch."""
        return self.cell.init_state(batch_size)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[MRNState] = None,
        return_sequences: bool = True,
        return_state: bool = False,
        return_activations: bool = False,
    ) -> Union[torch.Tensor, Tuple]:
        """
        Process a sequence.

        Args:
            inputs: [B, T, input_size] or [T, input_size]
            states: initial MRNState. If None, a fresh state is created with
                the right batch size. Pass an existing state to continue from
                a previous segment (BPTT will run through the full chain).
            return_sequences: if True, return outputs at every timestep,
                otherwise only the final output.
            return_state: also return the final state.
            return_activations: also return all per-layer activations.

        Returns:
            output (and optionally activations and final_state).
        """
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(0)
            squeeze_batch = True
        else:
            squeeze_batch = False

        batch_size, sequence_length, _ = inputs.shape

        if states is None:
            states = self.cell.init_state(batch_size=batch_size)

        outputs_values = []
        layer_activations_list = {i: [] for i in range(self.num_layers)}

        for t in range(sequence_length):
            cell_output, cell_activations, states = self.cell(inputs[:, t], states)
            outputs_values.append(cell_output)
            for layer_index, activation in cell_activations.items():
                layer_activations_list[layer_index].append(activation)

        outputs_values = torch.stack(outputs_values, dim=1)
        all_layer_activations = {
            layer_index: torch.stack(activations, dim=1)
            for layer_index, activations in layer_activations_list.items()
        }

        if squeeze_batch:
            outputs_values = outputs_values.squeeze(0)
            all_layer_activations = {
                k: v.squeeze(0) for k, v in all_layer_activations.items()
            }

        if not return_sequences:
            outputs_values = (
                outputs_values[:, -1] if not squeeze_batch else outputs_values[-1]
            )
            all_layer_activations = {
                k: (v[:, -1] if not squeeze_batch else v[-1])
                for k, v in all_layer_activations.items()
            }

        result = (outputs_values,)
        if return_activations:
            result = result + (all_layer_activations,)
        if return_state:
            result = result + (states,)
        return result if len(result) > 1 else result[0]

    def set_update_memory(self, update: bool):
        """Enable or disable memory updates during forward passes."""
        self.cell._update_memory_flag = update