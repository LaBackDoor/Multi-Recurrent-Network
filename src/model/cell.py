"""
Multi-Recurrent Neural Network (MRN) - Cell-Based Implementation

BROADCAST DESIGN: All memory banks project to ALL hidden layers
"""

from typing import Dict, List, Optional, Tuple, NamedTuple
import torch
import torch.nn as nn


class MRNState(NamedTuple):
    """Container for MRN memory state."""

    memory_banks: Dict[int, torch.Tensor]

    def clone(self) -> "MRNState":
        return MRNState(
            memory_banks={k: v.clone() for k, v in self.memory_banks.items()}
        )


class MemoryBank(nn.Module):
    """
    Memory bank that broadcasts to multiple target layers.

    Each memory bank has separate weight matrices for each hidden layer it feeds.
    """

    def __init__(
        self,
        num_items: int,
        layer_size: int,
        target_layer_sizes: Dict[int, int],  # {layer_idx: layer_size}
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.num_items = num_items
        self.layer_size = layer_size
        self.target_layer_sizes = target_layer_sizes
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Memory buffer
        self.register_buffer(
            "memory", torch.rand(num_items, layer_size, device=self.device)
        )

        # Separate weights for each target layer
        self.memory_weights = nn.ModuleDict()

        for target_layer_idx, target_size in target_layer_sizes.items():
            weights_for_layer = nn.ParameterList(
                [
                    nn.Parameter(
                        torch.rand(target_size, layer_size, device=self.device)
                    )
                    for _ in range(num_items)
                ]
            )
            self.memory_weights[str(target_layer_idx)] = weights_for_layer

    def compute_context(
        self, memory_snapshot: torch.Tensor, target_layer: int
    ) -> torch.Tensor:
        """
        Compute context for a specific target layer.

        Args:
            memory_snapshot: Current memory [num_items, layer_size]
            target_layer: Which layer to compute context for

        Returns:
            Context vector [target_layer_size]
        """
        target_layer_str = str(target_layer)

        if target_layer_str not in self.memory_weights:
            raise ValueError(f"No weights for target layer {target_layer}")

        weights = self.memory_weights[target_layer_str]
        context_parts = []

        for i in range(self.num_items):
            weight = weights[i]
            context_parts.append(torch.matmul(weight, memory_snapshot[i]))

        if context_parts:
            return torch.stack(context_parts).sum(dim=0)
        else:
            target_size = self.target_layer_sizes[target_layer]
            return torch.zeros(target_size, device=self.device)

    def update_memory(
        self, new_activation: torch.Tensor, memory_snapshot: torch.Tensor
    ) -> torch.Tensor:
        """Sluggish state space update."""
        new_memory = torch.zeros_like(memory_snapshot)
        for i in range(self.num_items):
            factor = (i + 1) / self.num_items
            new_memory[i] = factor * new_activation + (1 - factor) * memory_snapshot[i]
        return new_memory

    def get_memory_state(self) -> torch.Tensor:
        return self.memory.clone()

    def set_memory_state(self, memory: torch.Tensor) -> None:
        with torch.no_grad():
            self.memory.copy_(memory)


class MRNCell(nn.Module):
    """
    MRN Cell with BROADCAST memory design.

    All memory banks feed ALL hidden layers.
    The output layer remains memory-free.
    """

    def __init__(
        self,
        nn_structure: List[int],
        memory_structure: List[int],
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        if len(nn_structure) < 3:
            raise ValueError(
                f"nn_structure must have at least 3 layers, got {len(nn_structure)}"
            )

        self.nn_structure = nn_structure
        self.num_layers = len(nn_structure)
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.memory_structure = list(memory_structure) + [0] * (
            self.num_layers - len(memory_structure)
        )
        self._update_memory_flag = True

        self.input_size = nn_structure[0]
        self.output_size = nn_structure[-1]
        self.weights = nn.ParameterDict()
        self.biases = nn.ParameterDict()

        # Create network weights
        for layer_idx in range(1, self.num_layers):
            prev_size = nn_structure[layer_idx - 1]
            curr_size = nn_structure[layer_idx]

            self.weights[str(layer_idx)] = nn.Parameter(
                torch.rand(curr_size, prev_size, device=self.device)
            )
            self.biases[str(layer_idx)] = nn.Parameter(
                torch.rand(curr_size, device=self.device)
            )

        # Create memory banks - ALL project to ALL hidden layers
        self.memory_banks = nn.ModuleDict()

        # Build dict of hidden layer sizes (exclude input layer 0 and output layer)
        hidden_layer_sizes = {i: nn_structure[i] for i in range(1, self.num_layers - 1)}

        for layer_idx in range(self.num_layers):
            num_memories = self.memory_structure[layer_idx]

            if num_memories > 0:
                layer_size = nn_structure[layer_idx]

                self.memory_banks[str(layer_idx)] = MemoryBank(
                    num_items=num_memories,
                    layer_size=layer_size,
                    target_layer_sizes=hidden_layer_sizes,
                    device=self.device,
                )

    def forward(
        self, inputs: torch.Tensor, state: Optional[MRNState] = None
    ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor], MRNState]:
        """Process one timestep."""
        # Handle batch dimension
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size = inputs.shape[0]

        # Snapshot memory state
        memory_snapshots = {}
        if state is not None:
            memory_snapshots = {
                int(k): v.detach() for k, v in state.memory_banks.items()
            }
        else:
            for layer_idx, bank in self.memory_banks.items():
                memory_snapshots[int(layer_idx)] = bank.memory.detach().clone()

        # Store activations
        activations = {0: inputs}

        # Forward pass through all layers
        for layer_idx in range(1, self.num_layers):
            prev_activation = activations[layer_idx - 1]

            # Base computation: W @ prev + b
            activation_pre = (
                torch.matmul(prev_activation, self.weights[str(layer_idx)].t())
                + self.biases[str(layer_idx)]
            )

            # BROADCAST: Add memory context to ALL HIDDEN LAYERS
            if layer_idx < self.num_layers - 1:  # All hidden layers
                context_parts = []

                for mem_layer_idx_str, bank in self.memory_banks.items():
                    mem_layer_idx = int(mem_layer_idx_str)
                    if mem_layer_idx in memory_snapshots:
                        # Get context FOR THIS specific target layer
                        bank_context = bank.compute_context(
                            memory_snapshots[mem_layer_idx], target_layer=layer_idx
                        )
                        context_parts.append(
                            bank_context.unsqueeze(0).expand(batch_size, -1)
                        )

                if context_parts:
                    context = torch.stack(context_parts).sum(dim=0)
                    activation_pre = activation_pre + context

            # Apply activation function
            if layer_idx == self.num_layers - 1:
                activation = activation_pre  # Output: linear
            else:
                activation = torch.sigmoid(activation_pre)  # Hidden: sigmoid

            activations[layer_idx] = activation

        # Update memory banks
        new_memory_snapshots = {}
        if self._update_memory_flag:
            for layer_idx in range(self.num_layers):
                layer_idx_str = str(layer_idx)
                if layer_idx_str in self.memory_banks:
                    update_activation = activations[layer_idx][0].detach()
                    bank = self.memory_banks[layer_idx_str]

                    old_memory = memory_snapshots[layer_idx]
                    new_memory = bank.update_memory(update_activation, old_memory)
                    new_memory_snapshots[layer_idx] = new_memory

                    with torch.no_grad():
                        bank.memory.copy_(new_memory)
                else:
                    if layer_idx in memory_snapshots:
                        new_memory_snapshots[layer_idx] = memory_snapshots[layer_idx]
        else:
            new_memory_snapshots = memory_snapshots

        new_state = MRNState(memory_banks=new_memory_snapshots)
        output = activations[self.num_layers - 1]

        if squeeze_output:
            output = output.squeeze(0)
            activations = {k: v.squeeze(0) for k, v in activations.items()}

        return output, activations, new_state

    def init_state(self) -> MRNState:
        """Initialize a fresh state."""
        return MRNState(
            memory_banks={
                int(layer_idx): bank.get_memory_state()
                for layer_idx, bank in self.memory_banks.items()
            }
        )
