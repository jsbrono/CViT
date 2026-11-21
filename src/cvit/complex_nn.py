"""Complex-valued building blocks: activations, dropout, layer norms and weight initialisation.

PyTorch supports complex ``nn.Linear`` / ``nn.Conv2d`` layers natively (``dtype=torch.complex64``),
but not complex activations, dropout, normalisation or initialisation. Those are defined here.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class SplitActivation(nn.Module):
    """Applies a real activation to the real and imaginary parts independently."""

    def __init__(self, activation: nn.Module):
        super().__init__()
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.complex(self.activation(x.real), self.activation(x.imag))


class SplitReLU(SplitActivation):
    """``ReLU(Re z) + i ReLU(Im z)``, often called CReLU in the literature."""

    def __init__(self):
        super().__init__(nn.ReLU())


class SplitGELU(SplitActivation):
    """``GELU(Re z) + i GELU(Im z)``."""

    def __init__(self):
        super().__init__(nn.GELU())


class ComplexDropout(nn.Module):
    """Dropout for complex tensors.

    Each complex value is kept or dropped as a whole (its real and imaginary parts share one
    mask entry), and kept values are rescaled by ``1 / (1 - p)`` as in standard dropout.
    """

    def __init__(self, p: float = 0.1):
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError(f"dropout probability must be in [0, 1), got {p}")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        mask = F.dropout(torch.ones_like(x.real), self.p, training=True)
        return x * mask


class SplitLayerNorm(nn.Module):
    """Layer norm applied separately to the real and imaginary parts.

    Simple, but it ignores the correlation between the two parts.
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.norm_real = nn.LayerNorm(normalized_shape, eps=eps)
        self.norm_imag = nn.LayerNorm(normalized_shape, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.complex(self.norm_real(x.real), self.norm_imag(x.imag))


class ComplexLayerNorm(nn.Module):
    """Layer norm by 2D whitening over the last dimension.

    Adapted from complex batch normalisation (Trabelsi et al., 2018, "Deep Complex Networks";
    reference implementation: https://github.com/ChihebTrabelsi/deep_complex_networks). Each
    token is treated as a cloud of points in the 2D (real, imag) plane:

    1. centre it, ``x <- x - E[x]``;
    2. whiten it, ``x <- V^(-1/2) x``, where ``V`` is the 2x2 covariance of (Re x, Im x).
       The output parts then have unit variance and are uncorrelated;
    3. optionally apply a learned scale (a symmetric 2x2 matrix ``gamma``) and complex shift
       ``beta``.

    ``gamma`` starts as ``I / sqrt(2)`` so the output starts with ``E|x|^2 = 1``.
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-5, elementwise_affine: bool = True):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.gamma_rr = nn.Parameter(torch.full((normalized_shape,), 1 / math.sqrt(2)))
            self.gamma_ri = nn.Parameter(torch.zeros(normalized_shape))
            self.gamma_ii = nn.Parameter(torch.full((normalized_shape,), 1 / math.sqrt(2)))
            self.beta = nn.Parameter(torch.zeros(normalized_shape, dtype=torch.complex64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x - x.mean(dim=-1, keepdim=True)
        re, im = x.real, x.imag

        # Covariance matrix V = [[v_rr, v_ri], [v_ri, v_ii]]. Only the diagonal gets eps, which
        # keeps det(V) > 0 because v_ri^2 <= v_rr * v_ii (Cauchy-Schwarz).
        v_rr = (re * re).mean(dim=-1, keepdim=True) + self.eps
        v_ii = (im * im).mean(dim=-1, keepdim=True) + self.eps
        v_ri = (re * im).mean(dim=-1, keepdim=True)

        # Closed-form inverse square root of a 2x2 SPD matrix. With s = sqrt(det V) and
        # t = sqrt(trace V + 2s): sqrt(V) = (V + sI) / t, so V^(-1/2) = adj(V + sI) / (s t).
        s = torch.sqrt(v_rr * v_ii - v_ri * v_ri)
        t = torch.sqrt(v_rr + v_ii + 2 * s)
        inv_st = 1.0 / (s * t)
        w_rr = (v_ii + s) * inv_st
        w_ii = (v_rr + s) * inv_st
        w_ri = -v_ri * inv_st

        out_re = w_rr * re + w_ri * im
        out_im = w_ri * re + w_ii * im

        if self.elementwise_affine:
            out_re, out_im = (
                self.gamma_rr * out_re + self.gamma_ri * out_im + self.beta.real,
                self.gamma_ri * out_re + self.gamma_ii * out_im + self.beta.imag,
            )
        return torch.complex(out_re, out_im)


def complex_kaiming_(weight: torch.Tensor) -> torch.Tensor:
    """In-place complex He initialisation (Trabelsi et al., 2018).

    The modulus is drawn from a Rayleigh distribution with ``sigma = 1 / sqrt(fan_in)``, so that
    ``Var(w) = E|w|^2 = 2 / fan_in`` (the He criterion), and the phase uniformly from
    ``[-pi, pi)``. ``fan_in`` is ``in_features`` for linear layers and
    ``in_channels * kernel_h * kernel_w`` for convolutions.
    """
    if not weight.is_complex():
        raise TypeError(f"expected a complex tensor, got {weight.dtype}")
    fan_in = weight[0].numel()
    sigma = 1.0 / math.sqrt(fan_in)

    # Inverse-CDF sample of Rayleigh(sigma): sigma * sqrt(-2 ln(1 - U)), U ~ Uniform[0, 1).
    uniform = torch.rand(weight.shape, device=weight.device)
    modulus = sigma * torch.sqrt(-2.0 * torch.log1p(-uniform))
    phase = (2.0 * torch.rand(weight.shape, device=weight.device) - 1.0) * math.pi

    with torch.no_grad():
        weight.copy_(torch.polar(modulus, phase))
    return weight


def init_complex_weights(module: nn.Module) -> int:
    """Applies :func:`complex_kaiming_` to every complex ``weight`` in ``module``.

    Returns the number of tensors initialised.
    """
    count = 0
    for submodule in module.modules():
        weight = getattr(submodule, "weight", None)
        if isinstance(weight, nn.Parameter) and weight.is_complex():
            complex_kaiming_(weight)
            count += 1
    return count
