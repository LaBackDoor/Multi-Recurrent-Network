"""
Ratio Control Unit (RCU) for SL-MR-KAN.

The RCU is the v2 mechanism by which a memory bank's per-item layer-link
ratios are *learned* rather than fixed at (i + 1) / K. It maps the current
external input concatenated with this bank's flattened memory state to K
ratios in (0, 1). Self-link factors are derived as 1 - layer-link, matching
the thesis convention (Orojo 2022, Ch. 6).

Two backends:

- ``use_kan=True`` (default): the RCU is a small ``KANLinear``. This makes
  the ratio function itself a learnable univariate spline composition, which
  is the natural KAN analog of thesis SL-MRN 2 and is the flagship v2
  feature ("SL-MR-KAN").

- ``use_kan=False``: the RCU is a plain ``nn.Linear``. Useful for ablation:
  isolate "learnable ratios beat fixed ratios" from "KAN is the right
  parameterisation for the ratio function" by toggling this flag.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from src.model.kan.kan_linear import KANLinear


class RatioControlUnit(nn.Module):
    """Learn K layer-link ratios in (0, 1) from (current_input, memory).

    Args:
        input_size: width of the external input I_t.
        memory_size: total flattened size of this bank's memory, i.e.
            num_items * layer_size.
        num_items: K, the number of memory items in this bank. Equals the
            number of ratios produced.
        use_kan: True -> KANLinear backend, False -> nn.Linear backend.
            Toggling this lets you ablate KAN-vs-linear specifically for the
            ratio path while keeping the memory projections KAN-shaped.
        kan_grid_size, kan_spline_order, kan_use_layernorm, kan_grid_range:
            passed through to KANLinear when ``use_kan=True``. Defaults match
            the cell's defaults so the RCU shares stability properties with
            the rest of the network.
    """

    def __init__(
        self,
        input_size: int,
        memory_size: int,
        num_items: int,
        use_kan: bool = True,
        kan_grid_size: int = 3,
        kan_spline_order: int = 3,
        kan_enable_standalone_scale_spline: bool = False,
        kan_base_activation: type = nn.SiLU,
        kan_use_layernorm: bool = True,
        kan_grid_range: Optional[Tuple[float, float]] = None,
    ):
        super().__init__()
        self.input_size = input_size
        self.memory_size = memory_size
        self.num_items = num_items
        self.use_kan = use_kan

        in_features = input_size + memory_size

        if use_kan:
            self.unit = KANLinear(
                in_features=in_features,
                out_features=num_items,
                grid_size=kan_grid_size,
                spline_order=kan_spline_order,
                enable_standalone_scale_spline=kan_enable_standalone_scale_spline,
                base_activation=kan_base_activation,
                use_layernorm=kan_use_layernorm,
                grid_range=kan_grid_range,
            )
        else:
            self.unit = nn.Linear(in_features, num_items)

    def forward(
        self, current_input: torch.Tensor, memory: torch.Tensor
    ) -> torch.Tensor:
        """Compute K ratios in (0, 1) for one bank.

        Args:
            current_input: [B, input_size] - the external input I_t.
            memory:        [B, num_items, layer_size] - this bank's memory.

        Returns:
            ratios: [B, num_items], each in (0, 1).
        """
        flat_mem = memory.reshape(memory.size(0), -1)  # [B, K*L]
        x = torch.cat([current_input, flat_mem], dim=-1)  # [B, in_features]
        logits = self.unit(x)  # [B, K]
        return torch.sigmoid(logits)
