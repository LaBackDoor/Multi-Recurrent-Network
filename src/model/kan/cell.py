"""
MR-KAN cell.

v1: Canonical MRN structure (Orojo 2022, Chapter 3) with memory-to-hidden
projections replaced by KANLinear. Input->hidden and hidden->output remain plain
nn.Linear so the parameter budget stays tight and contributions attributable to
KAN are localised to the temporal-integration path.

v2 adds three orthogonal toggles, all default-off so v1 behavior is unchanged:

- ``learn_ratios``: SL-MR-KAN. Each bank gets a RatioControlUnit that learns
  per-item layer-link ratios from (current_input, this bank's memory). When
  False, the bank uses fixed ratios (i + 1) / K from the thesis canonical MRN.

- ``kan_input_path``: replace W_ih (input -> first hidden) with KANLinear.
  Lets you ablate "is KAN useful for the input path".

- ``kan_output_path``: replace W_ho (last hidden -> output) with KANLinear.
  The output-path KAN defaults to base_activation=Identity to preserve the
  thesis's "linear output layer" semantic; the spline branch then provides a
  learnable nonlinear correction on top of the linear residual.

Memory semantics (preserved):
- Sluggish update: M_t^{k,i} = r_i * L_{t-1}^k + (1 - r_i) * M_{t-1}^{k,i}.
  r_i is fixed (i+1)/K when learn_ratios=False, learned by RCU when True.
- Memory feeds the pre-activation of its target hidden layer; output is linear
  (or near-linear when kan_output_path=True with the default Identity base).

Chain topology for deeper networks follows the existing MRN README; for 3-layer
networks everything collapses to the canonical single hidden layer.
"""

from typing import Dict, List, Optional, Tuple, NamedTuple

import torch
import torch.nn as nn

from src.model.kan.kan_linear import KANLinear
from src.model.kan.ratio_control import RatioControlUnit


class MRKANState(NamedTuple):
    """Memory state for MR-KAN. Same layout as MRNState so the two are drop-in."""

    memory_banks: Dict[int, torch.Tensor]

    def clone(self) -> "MRKANState":
        return MRKANState(memory_banks={k: v.clone() for k, v in self.memory_banks.items()})

    def detach(self) -> "MRKANState":
        """Explicitly detach all tensors (breaks BPTT; use between windows)."""
        return MRKANState(
            memory_banks={k: v.detach() for k, v in self.memory_banks.items()}
        )


def _small_uniform(shape, scale: float, device: torch.device) -> torch.Tensor:
    return (torch.rand(*shape, device=device) * 2.0 - 1.0) * scale


class KANMemoryBank(nn.Module):
    """A bank of K memory items, each projected to one or more target hidden
    layers through its own KANLinear. The per-item ratios drive the sluggish
    update that forms the bank's state across time.

    Two ratio modes:
    - learn_ratios=False (v1, default): fixed r_i = (i + 1) / K.
    - learn_ratios=True (v2 SL-MR-KAN): an RCU computes r_i per timestep from
      (current_input, this bank's memory). See RatioControlUnit.

    Args:
        custom_ratios (optional): override the default (i+1)/K ratios. Used by
            Task 6's shrink() to preserve original ratios for surviving items
            after pruning.
        layer_link_ratios (registered buffer): per-item layer-link factors used
            in the non-RCU update path. Shape [num_items].

    Memory tensors live in MRKANState, not in this module, so memory is
    batch-aware and autograd can thread gradients through it.
    """

    def __init__(
        self,
        num_items: int,
        layer_size: int,
        target_layer_sizes: Dict[int, int],
        kan_grid_size: int = 3,
        kan_spline_order: int = 3,
        kan_enable_standalone_scale_spline: bool = False,
        kan_base_activation: type = nn.SiLU,
        kan_use_layernorm: bool = True,
        kan_grid_range: Optional[Tuple[float, float]] = None,
        init_memory_mode: str = "random",
        init_memory_value: float = 0.5,
        # v2 SL-MR-KAN args (default off)
        learn_ratios: bool = False,
        ratio_input_size: Optional[int] = None,
        ratio_control_use_kan: bool = True,
        # v2.4 custom ratios for surviving items after pruning
        custom_ratios: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        if init_memory_mode not in ("random", "constant"):
            raise ValueError(
                f"init_memory_mode must be 'random' or 'constant', got {init_memory_mode!r}"
            )
        if learn_ratios and ratio_input_size is None:
            raise ValueError(
                "learn_ratios=True requires ratio_input_size (the width of "
                "the external input I_t). The cell passes this through; "
                "callers constructing KANMemoryBank directly must supply it."
            )

        self.num_items = num_items
        self.layer_size = layer_size
        self.target_layer_sizes = dict(target_layer_sizes)
        self.init_memory_mode = init_memory_mode
        self.init_memory_value = init_memory_value
        self.learn_ratios = learn_ratios
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Memory -> target-layer projections. One KANLinear per (target, item)
        # so a future spline-shape pruning pass can compare items 1:1.
        self.memory_kans = nn.ModuleDict()
        for target_layer_idx, target_size in target_layer_sizes.items():
            per_item = nn.ModuleList(
                [
                    KANLinear(
                        in_features=layer_size,
                        out_features=target_size,
                        grid_size=kan_grid_size,
                        spline_order=kan_spline_order,
                        enable_standalone_scale_spline=kan_enable_standalone_scale_spline,
                        base_activation=kan_base_activation,
                        use_layernorm=kan_use_layernorm,
                        grid_range=kan_grid_range,
                    )
                    for _ in range(num_items)
                ]
            )
            self.memory_kans[str(target_layer_idx)] = per_item

        # Optional RCU (v2 SL-MR-KAN). When None, we use fixed (i+1)/K ratios.
        if learn_ratios and num_items > 0:
            self.rcu: Optional[RatioControlUnit] = RatioControlUnit(
                input_size=ratio_input_size,
                memory_size=num_items * layer_size,
                num_items=num_items,
                use_kan=ratio_control_use_kan,
                kan_grid_size=kan_grid_size,
                kan_spline_order=kan_spline_order,
                kan_enable_standalone_scale_spline=kan_enable_standalone_scale_spline,
                kan_base_activation=kan_base_activation,
                kan_use_layernorm=kan_use_layernorm,
                kan_grid_range=kan_grid_range,
            )
        else:
            self.rcu = None

        # Persist KAN config for shrink() surgery in Task 6 (used to construct
        # a new bank with matching configuration when pruning items).
        self._kan_grid_size = kan_grid_size
        self._kan_spline_order = kan_spline_order
        self._kan_enable_standalone_scale_spline = kan_enable_standalone_scale_spline
        self._kan_base_activation = kan_base_activation
        self._kan_use_layernorm = kan_use_layernorm
        self._kan_grid_range = kan_grid_range
        self._ratio_control_use_kan = ratio_control_use_kan

        # layer_link_ratios buffer: per-item layer-link factors for the
        # sluggish update. Default (i+1)/K (v1 formula); shrink() slices
        # this buffer when dropping items so survivors keep their original
        # ratios.
        if custom_ratios is None:
            ratios = torch.tensor(
                [(i + 1) / num_items for i in range(num_items)], dtype=torch.float32
            )
        else:
            if custom_ratios.shape != (num_items,):
                raise ValueError(
                    f"custom_ratios must have shape ({num_items},), got {tuple(custom_ratios.shape)}"
                )
            ratios = custom_ratios.detach().to(torch.float32).clone()
        self.register_buffer("layer_link_ratios", ratios)

        self.to(self.device)

    def init_memory(self, batch_size: int) -> torch.Tensor:
        """Initial memory tensor of shape [B, num_items, layer_size]."""
        shape = (batch_size, self.num_items, self.layer_size)
        if self.init_memory_mode == "random":
            return torch.rand(*shape, device=self.device)
        return torch.full(shape, self.init_memory_value, device=self.device)

    def compute_context(
        self, memory: torch.Tensor, target_layer: int
    ) -> torch.Tensor:
        """Project memory into a target hidden layer's pre-activation space.

        Args:
            memory: [B, num_items, layer_size]
            target_layer: which hidden layer consumes this context

        Returns:
            context: [B, target_size]
        """
        target_key = str(target_layer)
        if target_key not in self.memory_kans:
            raise ValueError(f"No KAN weights for target layer {target_layer}")

        per_item = self.memory_kans[target_key]
        target_size = self.target_layer_sizes[target_layer]
        batch_size = memory.shape[0]

        if self.num_items == 0:
            return torch.zeros(batch_size, target_size, device=memory.device)

        parts = []
        for i in range(self.num_items):
            parts.append(per_item[i](memory[:, i, :]))
        return torch.stack(parts, dim=0).sum(dim=0)

    def update_memory(
        self,
        new_activation: torch.Tensor,
        memory: torch.Tensor,
        current_input: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sluggish update.

        Two paths, controlled at construction time by ``learn_ratios``:

        - RCU is None (v1): per-item ratio r_i comes from the layer_link_ratios
          buffer (default: (i + 1) / K, but may be overridden via custom_ratios
          in __init__).
        - RCU is present (v2): r_i computed from (current_input, memory).

        Args:
            new_activation: [B, layer_size]
            memory:         [B, num_items, layer_size]
            current_input:  [B, ratio_input_size] - required when RCU is
                present, ignored otherwise.

        Returns:
            new_memory: [B, num_items, layer_size]
        """
        if self.rcu is None:
            # v1 path - now reads from buffer instead of computing (i+1)/K inline
            ratios = self.layer_link_ratios.view(1, -1, 1).to(memory.device)  # [1, K, 1]
            new_act_expanded = new_activation.unsqueeze(1).expand_as(memory)
            return ratios * new_act_expanded + (1.0 - ratios) * memory

        # v2 path: learned ratios (unchanged from before)
        if current_input is None:
            raise ValueError(
                "learn_ratios=True requires current_input to be passed to "
                "update_memory. The MRKANCell does this automatically."
            )
        ratios = self.rcu(current_input, memory)            # [B, K]
        ratios = ratios.unsqueeze(-1)                       # [B, K, 1]
        new_act_expanded = new_activation.unsqueeze(1).expand_as(memory)  # [B, K, L]
        return ratios * new_act_expanded + (1.0 - ratios) * memory


class MRKANCell(nn.Module):
    """MR-KAN cell with v1 + v2 features.

    All v2 features are orthogonal toggles defaulting to off, so the v1
    behavior (memory-only KAN, fixed ratios, plain linear input/output) is
    preserved when no v2 flag is set.
    """

    def __init__(
        self,
        nn_structure: List[int],
        memory_structure: List[int],
        weight_init_range: float = 0.01,
        hidden_bias_init_value: Optional[float] = 0.5,
        init_memory_mode: str = "random",
        init_memory_value: float = 0.5,
        # KAN config (shared across memory KANs and any path KANs)
        kan_grid_size: int = 3,
        kan_spline_order: int = 3,
        kan_enable_standalone_scale_spline: bool = False,
        kan_base_activation: type = nn.SiLU,
        kan_use_layernorm: bool = True,
        kan_grid_range: Optional[Tuple[float, float]] = None,
        # v2 ablation toggles (all default to v1 behavior)
        learn_ratios: bool = False,
        ratio_control_use_kan: bool = True,
        kan_input_path: bool = False,
        kan_output_path: bool = False,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        if len(nn_structure) < 3:
            raise ValueError(
                f"nn_structure must have at least 3 layers, got {len(nn_structure)}"
            )

        self.nn_structure = list(nn_structure)
        self.num_layers = len(nn_structure)
        self.weight_init_range = weight_init_range
        self.hidden_bias_init_value = hidden_bias_init_value
        self.init_memory_mode = init_memory_mode
        self.init_memory_value = init_memory_value
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.memory_structure = list(memory_structure) + [0] * (
            self.num_layers - len(memory_structure)
        )
        self._update_memory_flag = True

        # KAN config
        self.kan_grid_size = kan_grid_size
        self.kan_spline_order = kan_spline_order
        self.kan_enable_standalone_scale_spline = kan_enable_standalone_scale_spline
        self.kan_base_activation = kan_base_activation
        self.kan_use_layernorm = kan_use_layernorm
        self.kan_grid_range = kan_grid_range

        # v2 toggles
        self.learn_ratios = learn_ratios
        self.ratio_control_use_kan = ratio_control_use_kan
        self.kan_input_path = kan_input_path
        self.kan_output_path = kan_output_path

        self.input_size = nn_structure[0]
        self.output_size = nn_structure[-1]

        # Feedforward layers. Linear paths live in self.weights/self.biases
        # (matches the v1 layout exactly so the v1 parity test still works).
        # KAN-replaced paths live in self.kan_layers. Each layer index ends up
        # in exactly one of the two; forward picks the right one per index.
        self.weights = nn.ParameterDict()
        self.biases = nn.ParameterDict()
        self.kan_layers = nn.ModuleDict()

        vl = weight_init_range
        for layer_idx in range(1, self.num_layers):
            prev_size = nn_structure[layer_idx - 1]
            curr_size = nn_structure[layer_idx]
            is_first_hidden = layer_idx == 1
            is_output = layer_idx == self.num_layers - 1
            use_kan_path = (is_first_hidden and kan_input_path) or (
                is_output and kan_output_path
            )

            if use_kan_path:
                # Output-path KAN defaults to Identity base so the output stays
                # near-linear (preserving the canonical "linear output layer"
                # semantic). Input-path KAN uses the cell's configured base.
                path_base_activation = (
                    nn.Identity if is_output else kan_base_activation
                )
                self.kan_layers[str(layer_idx)] = KANLinear(
                    in_features=prev_size,
                    out_features=curr_size,
                    grid_size=kan_grid_size,
                    spline_order=kan_spline_order,
                    enable_standalone_scale_spline=kan_enable_standalone_scale_spline,
                    base_activation=path_base_activation,
                    use_layernorm=kan_use_layernorm,
                    grid_range=kan_grid_range,
                )
            else:
                self.weights[str(layer_idx)] = nn.Parameter(
                    _small_uniform((curr_size, prev_size), vl, self.device)
                )
                if is_first_hidden and hidden_bias_init_value is not None:
                    bias_tensor = torch.full(
                        (curr_size,), hidden_bias_init_value, device=self.device
                    )
                else:
                    bias_tensor = _small_uniform((curr_size,), vl, self.device)
                self.biases[str(layer_idx)] = nn.Parameter(bias_tensor)

        # Memory banks (chain topology, optional RCU).
        # Note: custom_ratios is not exposed here; Task 6's shrink() passes it
        # directly when rebuilding a pruned bank.
        self.memory_banks = nn.ModuleDict()
        for layer_idx in range(self.num_layers):
            num_memories = self.memory_structure[layer_idx]
            if num_memories > 0:
                target = self._chain_target(layer_idx)
                target_layer_sizes = {target: nn_structure[target]}
                self.memory_banks[str(layer_idx)] = KANMemoryBank(
                    num_items=num_memories,
                    layer_size=nn_structure[layer_idx],
                    target_layer_sizes=target_layer_sizes,
                    kan_grid_size=kan_grid_size,
                    kan_spline_order=kan_spline_order,
                    kan_enable_standalone_scale_spline=kan_enable_standalone_scale_spline,
                    kan_base_activation=kan_base_activation,
                    kan_use_layernorm=kan_use_layernorm,
                    kan_grid_range=kan_grid_range,
                    init_memory_mode=init_memory_mode,
                    init_memory_value=init_memory_value,
                    learn_ratios=learn_ratios,
                    ratio_input_size=self.input_size if learn_ratios else None,
                    ratio_control_use_kan=ratio_control_use_kan,
                    device=self.device,
                )

    def _chain_target(self, source_layer: int) -> int:
        """Chain topology rule (from the MRN README). For 3-layer networks
        this collapses to 'all memory feeds the single hidden layer'."""
        if source_layer == 0:
            return 1
        if source_layer == self.num_layers - 1:
            return 1
        if source_layer == self.num_layers - 2:
            return source_layer  # self-loop
        return source_layer + 1

    def init_state(self, batch_size: int = 1) -> MRKANState:
        return MRKANState(
            memory_banks={
                int(layer_idx): bank.init_memory(batch_size)
                for layer_idx, bank in self.memory_banks.items()
            }
        )

    def _layer_pre_activation(
        self, layer_idx: int, prev_activation: torch.Tensor
    ) -> torch.Tensor:
        """Apply this layer's feedforward transform: KAN if registered for
        this layer index, plain matmul + bias otherwise."""
        key = str(layer_idx)
        if key in self.kan_layers:
            return self.kan_layers[key](prev_activation)
        return (
            torch.matmul(prev_activation, self.weights[key].t()) + self.biases[key]
        )

    def forward(
        self, inputs: torch.Tensor, state: Optional[MRKANState] = None
    ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor], MRKANState]:
        """Process one timestep. Same shape contract as MRN.forward."""
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size = inputs.shape[0]

        if state is None:
            state = self.init_state(batch_size)

        memory = {int(k): v for k, v in state.memory_banks.items()}
        activations = {0: inputs}

        for layer_idx in range(1, self.num_layers):
            prev_activation = activations[layer_idx - 1]
            activation_pre = self._layer_pre_activation(layer_idx, prev_activation)

            # Memory context contributes only to hidden pre-activations
            # (canonical: output is memory-free).
            if layer_idx < self.num_layers - 1:
                context_parts = []
                for mem_layer_idx_str, bank in self.memory_banks.items():
                    mem_layer_idx = int(mem_layer_idx_str)
                    if (
                        mem_layer_idx in memory
                        and str(layer_idx) in bank.memory_kans
                    ):
                        bank_context = bank.compute_context(
                            memory[mem_layer_idx], target_layer=layer_idx
                        )
                        context_parts.append(bank_context)
                if context_parts:
                    context = torch.stack(context_parts, dim=0).sum(dim=0)
                    activation_pre = activation_pre + context

            if layer_idx == self.num_layers - 1:
                activation = activation_pre  # linear (or near-linear with Identity-base KAN)
            else:
                activation = torch.sigmoid(activation_pre)
            activations[layer_idx] = activation

        if self._update_memory_flag:
            new_memory = {}
            for layer_idx in range(self.num_layers):
                layer_idx_str = str(layer_idx)
                if layer_idx_str in self.memory_banks:
                    bank = self.memory_banks[layer_idx_str]
                    new_memory[layer_idx] = bank.update_memory(
                        activations[layer_idx],
                        memory[layer_idx],
                        current_input=inputs if bank.rcu is not None else None,
                    )
                elif layer_idx in memory:
                    new_memory[layer_idx] = memory[layer_idx]
        else:
            new_memory = memory

        new_state = MRKANState(memory_banks=new_memory)
        output = activations[self.num_layers - 1]

        if squeeze_output:
            output = output.squeeze(0)
            activations = {k: v.squeeze(0) for k, v in activations.items()}

        return output, activations, new_state

    def update_grids(self, calibration_inputs: Dict[int, torch.Tensor]) -> None:
        """Run efficient-kan's update_grid on every memory KAN using
        pre-collected calibration inputs.

        Use the higher-level ``MRKAN.calibrate_grids(loader)`` helper unless
        you need fine control. This method is exposed for back-compat with the
        v1 API.

        MUST run outside the training loop. update_grid is @torch.no_grad()
        and writes in place; calling it mid-sequence breaks BPTT.
        """
        for layer_idx_str, bank in self.memory_banks.items():
            mem_layer_idx = int(layer_idx_str)
            if mem_layer_idx not in calibration_inputs:
                continue
            x = calibration_inputs[mem_layer_idx]
            for per_item in bank.memory_kans.values():
                for kan in per_item:
                    kan.update_grid(x)
