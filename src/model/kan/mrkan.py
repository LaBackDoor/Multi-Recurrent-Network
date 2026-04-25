"""MR-KAN sequence wrapper. Identical call signature to MRN so the two are drop-in.

v2 adds three orthogonal toggles plus a ``calibrate_grids`` helper. All toggles
default to off so v1 behavior is preserved.
"""

from typing import Iterable, List, Optional, Tuple

import torch
from torch import nn

from src.model.kan.cell import MRKANCell, MRKANState
from src.model.kan.kan_linear import KANLinear


class MRKAN(nn.Module):
    """Multi-Recurrent KAN.

    Same sequence/batching/state semantics as MRN. The architectural change
    lives in the cell: memory->hidden projections are KANLinear; the input
    and output paths are plain Linear by default but can be swapped to KAN
    via the v2 ablation toggles below.

    v1 args:
        nn_structure, memory_structure: same as MRN.
        weight_init_range, hidden_bias_init_value: as in v1.
        init_memory_mode, init_memory_value: as in v1.
        kan_grid_size, kan_spline_order, kan_use_layernorm,
            kan_enable_standalone_scale_spline, kan_grid_range,
            kan_base_activation: KAN config shared across memory KANs and
            optional path KANs.

    v2 ablation toggles (all default to v1 behavior):
        learn_ratios: enable SL-MR-KAN. Each bank gets a RatioControlUnit
            that learns per-item layer-link ratios from (input, memory).
        ratio_control_use_kan: when learn_ratios=True, controls whether each
            RCU uses KANLinear (default) or plain nn.Linear. Lets you ablate
            "is KAN useful for ratios specifically".
        kan_input_path: replace W_ih (input -> first hidden) with KANLinear.
        kan_output_path: replace W_ho (last hidden -> output) with KANLinear
            (defaults base_activation=Identity to preserve linear-output
            semantics; spline branch provides nonlinear correction).
    """

    def __init__(
        self,
        nn_structure: List[int],
        memory_structure: List[int],
        weight_init_range: float = 0.01,
        hidden_bias_init_value: Optional[float] = 0.5,
        init_memory_mode: str = "random",
        init_memory_value: float = 0.5,
        kan_grid_size: int = 3,
        kan_spline_order: int = 3,
        kan_enable_standalone_scale_spline: bool = False,
        kan_base_activation: type = nn.SiLU,
        kan_use_layernorm: bool = True,
        kan_grid_range: Optional[Tuple[float, float]] = None,
        # v2 toggles
        learn_ratios: bool = False,
        ratio_control_use_kan: bool = True,
        kan_input_path: bool = False,
        kan_output_path: bool = False,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.nn_structure = nn_structure
        self.memory_structure = memory_structure
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.input_size = nn_structure[0]
        self.output_size = nn_structure[-1]
        self.num_layers = len(nn_structure)

        self.cell = MRKANCell(
            nn_structure=nn_structure,
            memory_structure=memory_structure,
            weight_init_range=weight_init_range,
            hidden_bias_init_value=hidden_bias_init_value,
            init_memory_mode=init_memory_mode,
            init_memory_value=init_memory_value,
            kan_grid_size=kan_grid_size,
            kan_spline_order=kan_spline_order,
            kan_enable_standalone_scale_spline=kan_enable_standalone_scale_spline,
            kan_base_activation=kan_base_activation,
            kan_use_layernorm=kan_use_layernorm,
            kan_grid_range=kan_grid_range,
            learn_ratios=learn_ratios,
            ratio_control_use_kan=ratio_control_use_kan,
            kan_input_path=kan_input_path,
            kan_output_path=kan_output_path,
            device=self.device,
        )

    def init_state(self, batch_size: int = 1) -> MRKANState:
        return self.cell.init_state(batch_size)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Optional[MRKANState] = None,
        return_sequences: bool = True,
        return_state: bool = False,
        return_activations: bool = False,
    ):
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
        self.cell._update_memory_flag = update

    def update_grids(self, calibration_inputs):
        """Forward to the cell. See MRKANCell.update_grids for the contract.

        Must be called OUTSIDE a forward/backward pass; it is @torch.no_grad()
        and in-place.
        """
        self.cell.update_grids(calibration_inputs)

    @torch.no_grad()
    def calibrate_grids(
        self,
        dataloader: Iterable,
        n_batches: int = 4,
        device: Optional[torch.device] = None,
    ) -> None:
        """v2: adapt every KANLinear's grid to typical training-data inputs.

        Pattern (run between epochs, never inside a training loop):

        1. Switch to eval, freeze memory updates.
        2. Hook every KANLinear and collect its forward inputs.
        3. Run ``n_batches`` forward passes on the dataloader.
        4. Call ``update_grid`` on each KANLinear with its collected samples.
        5. Restore train/memory-update modes.

        ``update_grid`` is ``@torch.no_grad()`` and writes in place to the
        knot grid and refits spline_weight via least-squares; this would
        break BPTT if triggered mid-sequence. Calling it via this helper is
        safe because the entire helper is also no-grad and runs outside any
        backward pass.

        Args:
            dataloader: any iterable yielding either ``Tensor`` batches or
                ``(inputs, ...)`` tuples whose first element is the batch
                input. Should yield batches of shape ``[B, T, input_size]``
                or ``[T, input_size]``, just like training.
            n_batches: how many batches to draw before refitting. Default 4
                gives a reasonable sample without long calibration runs.
            device: optionally move calibration inputs onto this device.
                Defaults to ``self.device``.
        """
        was_training = self.training
        device = device or self.device

        self.eval()
        self.set_update_memory(False)

        try:
            collected: dict[int, list[torch.Tensor]] = {}
            hooks = []

            def make_hook(kan_id: int):
                def hook(module, inputs, output):
                    x = inputs[0]
                    if x.dim() > 2:
                        x = x.reshape(-1, x.shape[-1])
                    collected.setdefault(kan_id, []).append(x.detach())

                return hook

            kan_modules: list[KANLinear] = []
            for module in self.modules():
                if isinstance(module, KANLinear):
                    kan_modules.append(module)
                    hooks.append(module.register_forward_hook(make_hook(id(module))))

            for i, batch in enumerate(dataloader):
                if i >= n_batches:
                    break
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                else:
                    x = batch
                x = x.to(device)
                self.forward(x)

            for h in hooks:
                h.remove()

            for module in kan_modules:
                samples = collected.get(id(module), [])
                if not samples:
                    continue
                x = torch.cat(samples, dim=0)
                # update_grid expects a 2D tensor [N, in_features]; our hook
                # already reshaped to that.
                module.update_grid(x)
        finally:
            self.set_update_memory(True)
            if was_training:
                self.train()
