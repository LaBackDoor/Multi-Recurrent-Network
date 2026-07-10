"""
Multi-Recurrent Neural Network (MRN) - Cell-Based Implementation

Changes vs previous version (see MRN_translation_audit_v2.md for rationale):

1. Weight init now uses uniform(-weight_init_range, weight_init_range) with a
   small default (0.01), matching the thesis ("weight initialisations are very
   small") and the NumPy reference. Was: torch.rand, i.e. uniform(0, 1), which
   saturated sigmoids at step 0.

2. BPTT is no longer blocked. Memory is carried in the MRNState tuple with full
   gradient connectivity, so autograd builds the temporal graph across timesteps.
   Was: every memory read and every update activation was .detach()'d, which
   truncated BPTT to 1 step.

3. Memory is batch-aware. MemoryBank no longer holds a single persistent buffer;
   memory tensors now have shape [B, num_items, layer_size] and every batch
   element updates its own memory. Was: one shared buffer, updated only from
   batch element [0].

4. hidden_bias_init_value exposes the NumPy 0.5*ones hidden bias as an explicit
   option (default: 0.5, matching NumPy). Set to None for small uniform init.

5. init_memory_mode controls initial memory values. "random" matches the thesis
   text ("memory randomly initialised"). "constant" matches the NumPy code
   (0.5*ones). Default: "random".

BROADCAST DESIGN: All memory banks project to ALL hidden layers. For 3-layer
networks this reduces to the single hidden layer, matching NumPy exactly.
"""

from contextlib import contextmanager
from typing import Dict, Iterator, List, Optional, Tuple, NamedTuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.device import default_device


@contextmanager
def sequence_weight_cache(banks: Iterator[nn.Module]) -> Iterator[None]:
    """Hold concatenated memory weights for the duration of one sequence.

    The concatenation in ``MemoryBank._concat_weight`` is loop-invariant: the
    parameters do not change across the timesteps of a single forward. Without
    a cache, an unrolled T-step forward rebuilds it T times, and backward then
    threads T separate cat nodes. Building it once per forward keeps a single
    cat node in the graph, so gradients accumulate into the same per-item
    parameters exactly as before.

    Cleared on exit so no autograd graph outlives the forward that built it.
    """
    entered = list(banks)
    for bank in entered:
        bank._weight_cache = {}
    try:
        yield
    finally:
        for bank in entered:
            bank._weight_cache = None


class MRNState(NamedTuple):
    """
    Container for MRN memory state.

    memory_banks[layer_idx] has shape [B, num_items, layer_size] where B is
    the batch size, num_items is the number of memory banks at that layer,
    and layer_size is the width of the layer whose activations are stored.
    """

    memory_banks: Dict[int, torch.Tensor]

    def clone(self) -> "MRNState":
        return MRNState(
            memory_banks={k: v.clone() for k, v in self.memory_banks.items()}
        )

    def detach(self) -> "MRNState":
        """Explicitly detach all tensors (breaks BPTT; use before a new window)."""
        return MRNState(
            memory_banks={k: v.detach() for k, v in self.memory_banks.items()}
        )


def _small_uniform(shape, scale: float, device: torch.device) -> torch.Tensor:
    """Uniform(-scale, scale) on the given shape."""
    return (torch.rand(*shape, device=device) * 2.0 - 1.0) * scale


class MemoryBank(nn.Module):
    """
    Memory bank with learned projections into one or more target hidden layers.

    No memory tensor is stored inside the module. Memory lives in MRNState so
    it is batch-aware and so autograd can thread gradients through it.
    """

    def __init__(
        self,
        num_items: int,
        layer_size: int,
        target_layer_sizes: Dict[int, int],
        weight_init_range: float = 0.01,
        init_memory_mode: str = "random",
        init_memory_value: float = 0.5,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.num_items = num_items
        self.layer_size = layer_size
        self.target_layer_sizes = dict(target_layer_sizes)
        self.init_memory_mode = init_memory_mode
        self.init_memory_value = init_memory_value
        self.device = device or default_device()
        self.use_fused_context = True
        # Populated only inside sequence_weight_cache(); None means "rebuild".
        self._weight_cache: Optional[Dict[int, torch.Tensor]] = None

        if init_memory_mode not in ("random", "constant"):
            raise ValueError(
                f"init_memory_mode must be 'random' or 'constant', "
                f"got {init_memory_mode!r}"
            )

        # Learned projection weights: one (target_size, layer_size) matrix per
        # (target layer, item) pair, initialized to small symmetric uniform.
        self.memory_weights = nn.ModuleDict()
        for target_layer_idx, target_size in target_layer_sizes.items():
            weights_for_layer = nn.ParameterList(
                [
                    nn.Parameter(
                        _small_uniform(
                            (target_size, layer_size), weight_init_range, self.device
                        )
                    )
                    for _ in range(num_items)
                ]
            )
            self.memory_weights[str(target_layer_idx)] = weights_for_layer

        # Sluggish-update ratios r_i = (i+1)/K, precomputed as [1, K, 1] so the
        # update is one broadcast instead of a K-step Python loop. Both r and
        # (1 - r) are materialised from the same float64 Python arithmetic the
        # loop used, so the fused update is bit-exact rather than 1-ulp off.
        # Non-persistent: they are derived from num_items, so keeping them out
        # of state_dict preserves checkpoint compatibility.
        ratios = [(i + 1) / num_items for i in range(num_items)] if num_items else []
        self.register_buffer(
            "_ratios",
            torch.tensor(ratios, dtype=torch.float32, device=self.device).view(1, -1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_one_minus_ratios",
            torch.tensor(
                [1.0 - r for r in ratios], dtype=torch.float32, device=self.device
            ).view(1, -1, 1),
            persistent=False,
        )

    def _concat_weight(self, target_layer: int) -> torch.Tensor:
        """Per-item weights concatenated along the input axis: [target, K*L].

        ``sum_i memory[:, i, :] @ W_i.T`` is exactly
        ``memory.reshape(B, K*L) @ cat(W_i, dim=1).T`` because the contraction
        over items and the contraction over layer_size are the same reduction.
        Item i occupies columns [i*L, (i+1)*L), matching the row-major flatten
        of memory's [B, K, L].
        """
        cache = self._weight_cache
        if cache is not None and target_layer in cache:
            return cache[target_layer]

        weights = self.memory_weights[str(target_layer)]
        concat = torch.cat([w for w in weights], dim=1)
        if cache is not None:
            cache[target_layer] = concat
        return concat

    def init_memory(self, batch_size: int) -> torch.Tensor:
        """
        Create an initial memory tensor of shape [B, num_items, layer_size].

        Per self.init_memory_mode: "random" (uniform [0, 1), matches the thesis
        text) or "constant" (filled with init_memory_value, matches NumPy).
        """
        # Follow the module's live parameter device rather than the
        # construction-time self.device, so `model.to(device)` keeps working.
        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = self.device
        shape = (batch_size, self.num_items, self.layer_size)
        if self.init_memory_mode == "random":
            return torch.rand(*shape, device=device)
        return torch.full(shape, self.init_memory_value, device=device)

    def compute_context(
        self, memory: torch.Tensor, target_layer: int
    ) -> torch.Tensor:
        """
        Compute the memory context for a specific target hidden layer.

        Args:
            memory: [B, num_items, layer_size]
            target_layer: which hidden layer is consuming this context

        Returns:
            context: [B, target_size]
        """
        target_layer_str = str(target_layer)
        if target_layer_str not in self.memory_weights:
            raise ValueError(f"No weights for target layer {target_layer}")

        weights = self.memory_weights[target_layer_str]
        target_size = self.target_layer_sizes[target_layer]
        batch_size = memory.shape[0]

        if self.num_items == 0:
            return torch.zeros(
                batch_size, target_size, device=memory.device, dtype=memory.dtype
            )

        if not self.use_fused_context:
            context_parts = []
            for i in range(self.num_items):
                # memory[:, i, :]: [B, layer_size]
                # weights[i]:      [target_size, layer_size]
                # result:          [B, target_size]
                context_i = torch.matmul(memory[:, i, :], weights[i].t())
                context_parts.append(context_i)
            return torch.stack(context_parts, dim=0).sum(dim=0)

        # Fused: one GEMM over the flattened items instead of K matmuls plus a
        # stack and a sum. The item reduction becomes part of the GEMM's own
        # inner-product reduction.
        return F.linear(memory.reshape(batch_size, -1), self._concat_weight(target_layer))

    def update_memory(
        self, new_activation: torch.Tensor, memory: torch.Tensor
    ) -> torch.Tensor:
        """
        Sluggish state-space update. For item i with ratio r_i = (i+1)/K:

            new_memory[:, i, :] = r_i * new_activation + (1 - r_i) * memory[:, i, :]

        Args:
            new_activation: [B, layer_size]
            memory:         [B, num_items, layer_size]

        Returns:
            new_memory:     [B, num_items, layer_size]
        """
        if self.num_items == 0:
            return memory
        return (
            self._ratios * new_activation.unsqueeze(1) + self._one_minus_ratios * memory
        )


class MRNCell(nn.Module):
    """
    MRN Cell with BROADCAST memory design.

    All memory banks feed ALL hidden layers. The output layer is memory-free.
    Memory is carried in MRNState; this module holds only parameters.
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
        """
        Args:
            nn_structure: layer sizes, e.g. [input, hidden, output]. Minimum 3.
            memory_structure: memories per layer. Auto-padded with zeros to
                match nn_structure length.
            weight_init_range: all feedforward and memory-projection weights
                are initialized to uniform(-weight_init_range, weight_init_range).
                Thesis uses a small range; NumPy uses 0.01.
            hidden_bias_init_value: if set, bias of the first hidden layer is
                initialized to a constant of this value (NumPy does 0.5).
                If None, first-hidden bias also uses small uniform.
            init_memory_mode: "random" (thesis text) or "constant" (NumPy).
            init_memory_value: value used when init_memory_mode == "constant".
            device: target device.
        """
        super().__init__()

        if len(nn_structure) < 3:
            raise ValueError(
                f"nn_structure must have at least 3 layers, got {len(nn_structure)}"
            )
        if len(memory_structure) > len(nn_structure):
            raise ValueError(
                f"memory_structure has {len(memory_structure)} entries but "
                f"nn_structure only has {len(nn_structure)} layers; the extra "
                f"entries would be silently ignored"
            )

        self.nn_structure = list(nn_structure)
        self.num_layers = len(nn_structure)
        self.weight_init_range = weight_init_range
        self.hidden_bias_init_value = hidden_bias_init_value
        self.init_memory_mode = init_memory_mode
        self.init_memory_value = init_memory_value
        self.device = device or default_device()
        self.memory_structure = list(memory_structure) + [0] * (
            self.num_layers - len(memory_structure)
        )
        self._update_memory_flag = True

        self.input_size = nn_structure[0]
        self.output_size = nn_structure[-1]
        self.weights = nn.ParameterDict()
        self.biases = nn.ParameterDict()

        vl = weight_init_range

        # Feedforward weights and biases
        for layer_idx in range(1, self.num_layers):
            prev_size = nn_structure[layer_idx - 1]
            curr_size = nn_structure[layer_idx]

            self.weights[str(layer_idx)] = nn.Parameter(
                _small_uniform((curr_size, prev_size), vl, self.device)
            )

            if layer_idx == 1 and hidden_bias_init_value is not None:
                bias_tensor = torch.full(
                    (curr_size,), hidden_bias_init_value, device=self.device
                )
            else:
                bias_tensor = _small_uniform((curr_size,), vl, self.device)
            self.biases[str(layer_idx)] = nn.Parameter(bias_tensor)

        # Memory banks: chain topology (Option C.1). Each bank projects to
        # exactly one hidden layer per the rule below. For 3-layer networks
        # all banks target layer 1 (the single hidden), recovering the
        # canonical MRN. See _chain_target for the rule.
        self.memory_banks = nn.ModuleDict()

        for layer_idx in range(self.num_layers):
            num_memories = self.memory_structure[layer_idx]
            if num_memories > 0:
                target = self._chain_target(layer_idx)
                target_layer_sizes = {target: nn_structure[target]}
                self.memory_banks[str(layer_idx)] = MemoryBank(
                    num_items=num_memories,
                    layer_size=nn_structure[layer_idx],
                    target_layer_sizes=target_layer_sizes,
                    weight_init_range=vl,
                    init_memory_mode=init_memory_mode,
                    init_memory_value=init_memory_value,
                    device=self.device,
                )

    def _chain_target(self, source_layer: int) -> int:
        """
        Chain topology rule: which hidden layer does memory at source_layer feed?

        - Input memory (layer 0) feeds the first hidden layer (1).
        - Hidden memory at layer L for 1 <= L <= num_layers-3 feeds layer L+1
          (the next hidden layer in the forward chain).
        - Last hidden's memory (layer num_layers-2) self-loops, since the next
          forward layer would be the output and the canonical MRN keeps the
          output memory-free.
        - Output memory (layer num_layers-1) feeds the first hidden layer (1),
          preserving the long-range feedback semantics of the canonical MRN
          where output memory feeds the input-side of the network.

        For 3-layer networks (num_layers=3) all of the above resolve to layer 1,
        so this rule reduces to the canonical "all memory feeds the hidden
        layer" topology and the implementation matches Ulbricht/Orojo exactly.
        """
        if source_layer == 0:
            return 1
        if source_layer == self.num_layers - 1:
            return 1
        if source_layer == self.num_layers - 2:
            return source_layer  # self-loop
        return source_layer + 1

    def init_state(self, batch_size: int = 1) -> MRNState:
        """Create a fresh initial MRNState for the given batch size."""
        return MRNState(
            memory_banks={
                int(layer_idx): bank.init_memory(batch_size)
                for layer_idx, bank in self.memory_banks.items()
            }
        )

    def set_fused_context(self, enabled: bool) -> None:
        """Toggle the fused memory-context path on every bank.

        Fused (default) contracts all items of a bank in a single GEMM; the
        loop fallback does one matmul per item. Both compute the same function
        - the fallback exists for debugging and A/B checks.
        """
        for bank in self.memory_banks.values():
            bank.use_fused_context = enabled

    def forward(
        self, inputs: torch.Tensor, state: Optional[MRNState] = None
    ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor], MRNState]:
        """
        Process one timestep.

        Args:
            inputs: [B, input_size] or [input_size]
            state: MRNState to read memory from. If None, a fresh state is
                created. Pass an existing state (from a previous forward) to
                chain timesteps together; autograd will backprop across them.

        Returns:
            output: [B, output_size] (or [output_size] if input was unbatched)
            activations: dict of per-layer activations at this timestep
            new_state: updated MRNState
        """
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size = inputs.shape[0]

        if state is None:
            state = self.init_state(batch_size)

        # Read memory from state WITHOUT detaching so BPTT works.
        memory = {int(k): v for k, v in state.memory_banks.items()}

        activations = {0: inputs}

        # Forward pass through all layers
        for layer_idx in range(1, self.num_layers):
            prev_activation = activations[layer_idx - 1]
            activation_pre = (
                torch.matmul(prev_activation, self.weights[str(layer_idx)].t())
                + self.biases[str(layer_idx)]
            )

            # Add memory context to every hidden layer (output is memory-free).
            # In chain topology, each bank projects to exactly one hidden layer,
            # so we skip banks that don't have weights for this target.
            if layer_idx < self.num_layers - 1:
                context_parts = []
                for mem_layer_idx_str, bank in self.memory_banks.items():
                    mem_layer_idx = int(mem_layer_idx_str)
                    if mem_layer_idx in memory and str(layer_idx) in bank.memory_weights:
                        bank_context = bank.compute_context(
                            memory[mem_layer_idx], target_layer=layer_idx
                        )
                        context_parts.append(bank_context)

                if context_parts:
                    context = torch.stack(context_parts, dim=0).sum(dim=0)
                    activation_pre = activation_pre + context

            if layer_idx == self.num_layers - 1:
                activation = activation_pre  # linear output
            else:
                activation = torch.sigmoid(activation_pre)

            activations[layer_idx] = activation

        # Update memory banks (no detach; gradients flow into next timestep)
        if self._update_memory_flag:
            new_memory = {}
            for layer_idx in range(self.num_layers):
                layer_idx_str = str(layer_idx)
                if layer_idx_str in self.memory_banks:
                    bank = self.memory_banks[layer_idx_str]
                    new_memory[layer_idx] = bank.update_memory(
                        activations[layer_idx], memory[layer_idx]
                    )
                elif layer_idx in memory:
                    new_memory[layer_idx] = memory[layer_idx]
        else:
            new_memory = memory

        new_state = MRNState(memory_banks=new_memory)
        output = activations[self.num_layers - 1]

        if squeeze_output:
            output = output.squeeze(0)
            activations = {k: v.squeeze(0) for k, v in activations.items()}

        return output, activations, new_state