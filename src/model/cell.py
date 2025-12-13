"""
Multi-Recurrent Neural Network (MRN) - Cell-Based Implementation

This implementation follows PyTorch conventions with a clean separation between:
- MRNCell: Processes a single timestep
- MRN: Processes entire sequences using MRNCell
- MRNState: Manages memory bank state
"""

from typing import Dict, List, Optional, Tuple, NamedTuple

import torch
import torch.nn as nn


class MRNState(NamedTuple):
    """
    Container for MRN memory state.

    Attributes:
        memory_banks: Dictionary mapping layer index to memory tensor
                     Shape: {layer_idx: [num_memory_items, layer_size]}
    """
    memory_banks: Dict[int, torch.Tensor]

    def clone(self) -> 'MRNState':
        """Create a deep copy of the state."""
        return MRNState(
            memory_banks={k: v.clone() for k, v in self.memory_banks.items()}
        )


class MemoryBank(nn.Module):
    """
    Manages a single memory bank for one layer.

    The memory bank creates a 'sluggish state space' where each memory item
    has different retention characteristics:
    - Lower indices (i=1): More flexible, favor recent information
    - Higher indices (i=n): More rigid, preserve historical information

    Args:
        num_items: Number of memory items in this bank
        layer_size: Dimension of the layer this memory bank belongs to
        output_size: Size of the output that uses this memory
        device: Device to create tensors on
    """

    def __init__(
        self,
        num_items: int,
        layer_size: int,
        output_size: int,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.num_items = num_items
        self.layer_size = layer_size
        self.output_size = output_size
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Memory items - these are updated manually, not by gradients
        self.register_buffer(
            'memory',
            torch.rand(num_items, layer_size, device=self.device)
        )

        # Memory weights project each memory item to the output
        # Shape: [num_items, output_size, layer_size]
        self.memory_weights = nn.ParameterList([
            nn.Parameter(torch.rand(output_size, layer_size, device=self.device))
            for _ in range(num_items)
        ])

    def compute_context(self, memory_snapshot: torch.Tensor) -> torch.Tensor:
        """
        Compute context vector from memory snapshot.

        Args:
            memory_snapshot: Snapshot of memory state [num_items, layer_size]

        Returns:
            Context tensor of shape [output_size]
        """
        # Use the provided snapshot instead of self.memory
        context_parts: list[torch.Tensor] = []
        for i, weight in enumerate(self.memory_weights):
            weight: torch.nn.Parameter
            context_parts.append(torch.matmul(weight, memory_snapshot[i]))

        # Sum without in-place operations
        if context_parts:
            context = torch.stack(context_parts).sum(dim=0)
        else:
            context = torch.zeros(self.output_size, device=self.device)
        return context

    def update_memory(self, new_activation: torch.Tensor, memory_snapshot: torch.Tensor) -> torch.Tensor:
        """
        Compute updated memory based on sluggish state space dynamics.

        Returns the new memory state without modifying the buffer.
        The buffer will be updated separately after all computations are done.

        For each memory item i (1-indexed):
            memory[i] = (i/n) * new_activation + (1 - i/n) * memory[i]

        Args:
            new_activation: New activation from the layer [layer_size]
            memory_snapshot: Current memory state [num_items, layer_size]

        Returns:
            Updated memory state [num_items, layer_size]
        """
        new_memory = torch.zeros_like(memory_snapshot)
        for i in range(self.num_items):
            factor = (i + 1) / self.num_items
            new_memory[i] = factor * new_activation + (1 - factor) * memory_snapshot[i]
        return new_memory

    def get_memory_state(self) -> torch.Tensor:
        """Return the current memory state."""
        return self.memory.clone()

    def set_memory_state(self, memory: torch.Tensor) -> None:
        """Set memory state from external tensor."""
        with torch.no_grad():
            self.memory.copy_(memory)


class MRNCell(nn.Module):
    """
    Multi-Recurrent Neural Network Cell - processes one timestep.

    This implements the core MRN computation for a single timestep with
    support for variable depth (minimum 3 layers).

    Args:
        nn_structure: List of layer sizes [input_size, hidden1_size, ..., output_size]
                     Minimum length is 3 (input, 1 hidden, output)
        memory_structure: List of memory bank sizes for each layer
                         If shorter than nn_structure, pads with zeros
                         Example: [4, 3, 0, 2] means 4 memories at input, 3 at first hidden,
                                 0 at second hidden, 2 at output
        device: Device to create tensors on
    """

    def __init__(
        self,
        nn_structure: List[int],
        memory_structure: List[int],
        device: Optional[torch.device] = None
    ):
        super().__init__()

        if len(nn_structure) < 3:
            raise ValueError(f"nn_structure must have at least 3 layers (input + hidden + output), got {len(nn_structure)}")

        self.nn_structure = nn_structure
        self.num_layers = len(nn_structure)
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.memory_structure = list(memory_structure) + [0] * (self.num_layers - len(memory_structure))

        # Flag to control memory updates (default True)
        self._update_memory_flag = True

        self.input_size = nn_structure[0]
        self.output_size = nn_structure[-1]
        self.weights = nn.ParameterDict()
        self.biases = nn.ParameterDict()

        for layer_idx in range(1, self.num_layers):
            prev_size = nn_structure[layer_idx - 1]
            curr_size = nn_structure[layer_idx]

            self.weights[str(layer_idx)] = nn.Parameter(
                torch.rand(curr_size, prev_size, device=self.device)
            )
            self.biases[str(layer_idx)] = nn.Parameter(
                torch.rand(curr_size, device=self.device)
            )

        # Create memory banks indexed by layer (0-indexed for input layer, etc.)
        # ALL memory banks project to the FIRST HIDDEN LAYER (layer 1)
        self.memory_banks = nn.ModuleDict()

        first_hidden_size = nn_structure[1]

        for layer_idx in range(self.num_layers):
            num_memories = self.memory_structure[layer_idx]

            if num_memories > 0:
                layer_size = nn_structure[layer_idx]

                self.memory_banks[str(layer_idx)] = MemoryBank(
                    num_items=num_memories,
                    layer_size=layer_size,
                    output_size=first_hidden_size,
                    device=self.device
                )

    def forward(
        self,
        inputs: torch.Tensor,
        state: Optional[MRNState] = None
    ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor], MRNState]:
        """
        Process one timestep through all layers.

        Args:
            inputs: Input tensor [batch_size, input_size] or [input_size]
            state: Previous MRN state (if None, uses current memory bank state)

        Returns:
            output: Output tensor [batch_size, output_size] or [output_size]
            activations: Dictionary mapping layer index to activation tensor
            new_state: Updated MRN state
        """
        # Handle batch dimension
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size = inputs.shape[0]

        # Snapshot memory state BEFORE any computation, we use this snapshot for the entire forward pass
        # and only compute the new state, never modifying buffers during forward
        memory_snapshots = {}
        if state is not None:
            # Use provided state
            memory_snapshots = {int(k): v.detach() for k, v in state.memory_banks.items()}
        else:
            # Use the current buffer state
            for layer_idx, bank in self.memory_banks.items():
                memory_snapshots[int(layer_idx)] = bank.memory.detach().clone()

        # Store activations for each layer
        activations = {0: inputs}

        # Forward pass through all layers
        for layer_idx in range(1, self.num_layers):
            prev_activation = activations[layer_idx - 1]

            # Get base computation: W @ prev + b
            activation_pre = torch.matmul(
                prev_activation,
                self.weights[str(layer_idx)].t()
            ) + self.biases[str(layer_idx)]

            # For the first hidden layer, add memory context
            if layer_idx == 1:
                # Collect context from all memory banks using snapshots
                context_parts = []
                for mem_layer_idx_str, bank in self.memory_banks.items():
                    mem_layer_idx = int(mem_layer_idx_str)
                    if mem_layer_idx in memory_snapshots:
                        bank_context = bank.compute_context(memory_snapshots[mem_layer_idx])
                        context_parts.append(bank_context.unsqueeze(0).expand(batch_size, -1))

                # Sum all context contributions
                if context_parts:
                    context = torch.stack(context_parts).sum(dim=0)
                else:
                    context = torch.zeros(batch_size, self.nn_structure[layer_idx], device=self.device)

                activation_pre = activation_pre + context

            # Apply activation function
            if layer_idx == self.num_layers - 1:
                # Output layer: linear
                activation = activation_pre
            else:
                # Hidden layers: sigmoid
                activation = torch.sigmoid(activation_pre)

            activations[layer_idx] = activation

        # Compute the new memory state
        new_memory_snapshots = {}
        if self._update_memory_flag:
            for layer_idx in range(self.num_layers):
                layer_idx_str = str(layer_idx)
                if layer_idx_str in self.memory_banks:
                    # Use first item in batch for memory update
                    update_activation = activations[layer_idx][0].detach()
                    bank = self.memory_banks[layer_idx_str]

                    old_memory = memory_snapshots[layer_idx]
                    new_memory = bank.update_memory(update_activation, old_memory)
                    new_memory_snapshots[layer_idx] = new_memory

                    # Update the actual buffer this happens after all forward computation is done
                    with torch.no_grad():
                        bank.memory.copy_(new_memory)
                else:
                    # No memory bank for this layer
                    if layer_idx in memory_snapshots:
                        new_memory_snapshots[layer_idx] = memory_snapshots[layer_idx]
        else:
            # If not updating, keep the old memory
            new_memory_snapshots = memory_snapshots

        # Create a new state from computed memory
        new_state = MRNState(memory_banks=new_memory_snapshots)

        # Get output
        output = activations[self.num_layers - 1]

        if squeeze_output:
            output = output.squeeze(0)
            activations = {k: v.squeeze(0) for k, v in activations.items()}

        return output, activations, new_state

    def init_state(self) -> MRNState:
        """
        Initialize a fresh MRN state.

        Returns:
            Initial MRN state
        """
        return MRNState(
            memory_banks={
                int(layer_idx): bank.get_memory_state()
                for layer_idx, bank in self.memory_banks.items()
            }
        )