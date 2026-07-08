"""
Vendored KANLinear for MR-KAN.

Source: https://github.com/Blealtan/efficient-kan (MIT License)
Original author: Blealtan. Vendored verbatim with two additions:

1. Optional LayerNorm applied to the flattened input before the base / spline
   branches. This is the stability trick TKAN uses under recurrent feedback.

2. ``grid_range`` is coupled to ``use_layernorm``. When LayerNorm is enabled and
   no grid_range is supplied, the default widens from (-1, 1) to (-3, 3) so that
   ~99.7% of a post-LayerNorm N(0, 1) sample falls inside the active grid (vs.
   ~68% at the narrower default). See the MR-KAN plan, Section 1.2.

Defaults are tuned for recurrent use: grid_size=3, spline_order=3, and
``enable_standalone_scale_spline=False`` to keep the parameter footprint near
~7x a plain nn.Linear (6 spline coefficients + 1 base weight per edge) rather
than the ~10x of efficient-kan's stock defaults.

``update_grid`` is preserved from the original but is not wired into forward;
calling it mid-sequence breaks BPTT because of the @torch.no_grad() decorator
and in-place grid / spline_weight writes. Call it only between full-sequence
forward+backward passes.
"""

from typing import Optional, Tuple

import math
import torch
import torch.nn.functional as F


class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 3,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        enable_standalone_scale_spline: bool = False,
        base_activation: type = torch.nn.SiLU,
        grid_eps: float = 0.02,
        grid_range: Optional[Tuple[float, float]] = None,
        use_layernorm: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        # LayerNorm over a single feature collapses to zero deterministically
        # (mean and variance are both taken over one element, so (x - mean)/std
        # always evaluates to ~0). This kills the base branch and blocks BPTT
        # through the spline on 1-dim source layers (e.g. regression-output
        # memory). Silently disable LN in that case.
        effective_use_layernorm = bool(use_layernorm) and in_features > 1
        self.use_layernorm = effective_use_layernorm

        # Couple grid_range to LayerNorm: post-LayerNorm samples span ~N(0, 1),
        # so a grid of (-3, 3) captures ~99.7% of them. Without LayerNorm fall
        # back to efficient-kan's original (-1, 1). The coupling keys off the
        # *effective* LN state so a 1-dim KANLinear requested with LN still
        # gets the narrower default it actually needs.
        if grid_range is None:
            grid_range = (-3.0, 3.0) if effective_use_layernorm else (-1.0, 1.0)
        self.grid_range = tuple(grid_range)

        h = (self.grid_range[1] - self.grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + self.grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.empty(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.empty(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        if effective_use_layernorm:
            self.layer_norm = torch.nn.LayerNorm(in_features)
        else:
            self.layer_norm = None

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(
            self.base_weight, a=math.sqrt(5) * self.scale_base
        )
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 0.5
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (
                    self.scale_spline
                    if not self.enable_standalone_scale_spline
                    else 1.0
                )
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                torch.nn.init.kaiming_uniform_(
                    self.spline_scaler, a=math.sqrt(5) * self.scale_spline
                )

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """Cox-de Boor recursion. x: (B, in_features) -> (B, in, grid_size + spline_order)."""
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid = self.grid  # (in, grid_size + 2*spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )
        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Fit coefficients so the spline interpolates (x, y). Used only at init
        and inside update_grid. Do not call during training forward."""
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(0, 1)  # (in, B, coeff)
        B = y.transpose(0, 1)                  # (in, B, out)
        solution = torch.linalg.lstsq(A, B).solution  # (in, coeff, out)
        result = solution.permute(2, 0, 1)     # (out, in, coeff)
        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self) -> torch.Tensor:
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output
        return output.reshape(*original_shape[:-1], self.out_features)

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin: float = 0.01) -> None:
        """Adapt the knot grid to the distribution of ``x``.

        WARNING: this method is ``@torch.no_grad()`` and writes in place. Calling
        it inside an unrolled sequence will break BPTT. Use it only BETWEEN full
        forward+backward passes (e.g. once per epoch on a calibration batch).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        if self.layer_norm is not None:
            # Match forward: the grid should track the post-LayerNorm distribution.
            x = self.layer_norm(x)

        batch = x.size(0)

        splines = self.b_splines(x)                       # (B, in, coeff)
        splines = splines.permute(1, 0, 2)                # (in, B, coeff)
        orig_coeff = self.scaled_spline_weight            # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)          # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, B, out)
        unreduced_spline_output = unreduced_spline_output.permute(1, 0, 2)  # (B, in, out)

        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(
                    self.spline_order, 0, -1, device=x.device
                ).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(
                    1, self.spline_order + 1, device=x.device
                ).unsqueeze(1),
            ],
            dim=0,
        )

        # (Near-)constant features - or fp32 rounding at large magnitudes -
        # can leave adjacent knots identical, which zeroes the Cox-de Boor
        # denominators in b_splines and floods the forward pass with NaNs.
        # Rebuild the grid from consecutive spacings clamped to a scale-aware
        # minimum so knots stay strictly increasing.
        finfo = torch.finfo(grid.dtype)
        scale = grid.abs().amax(dim=0, keepdim=True)
        min_spacing = torch.clamp(16.0 * finfo.eps * scale, min=1e-6)
        spacing = torch.maximum(grid.diff(dim=0), min_spacing)
        grid = torch.concatenate(
            [grid[:1], grid[:1] + torch.cumsum(spacing, dim=0)], dim=0
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(
        self, regularize_activation: float = 1.0, regularize_entropy: float = 1.0
    ) -> torch.Tensor:
        """Author's fused L1 + entropy regularizer (not the paper's sample-wise L1)."""
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )
