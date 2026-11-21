import math

import pytest
import torch
from torch import nn

from cvit.complex_nn import (
    ComplexDropout,
    ComplexLayerNorm,
    SplitGELU,
    SplitLayerNorm,
    SplitReLU,
    complex_kaiming_,
    init_complex_weights,
)


def randc(*shape: int) -> torch.Tensor:
    return torch.randn(*shape, dtype=torch.complex64)


@pytest.mark.parametrize(("act", "fn"), [(SplitReLU(), torch.relu), (SplitGELU(), nn.GELU())])
def test_split_activation_acts_on_each_part(act, fn):
    x = randc(4, 8)
    out = act(x)
    torch.testing.assert_close(out.real, fn(x.real))
    torch.testing.assert_close(out.imag, fn(x.imag))


def test_dropout_is_identity_in_eval_mode():
    drop = ComplexDropout(0.5).eval()
    x = randc(16, 16)
    assert torch.equal(drop(x), x)


def test_dropout_drops_whole_values_and_rescales():
    torch.manual_seed(0)
    drop = ComplexDropout(0.25).train()
    x = torch.ones(200, 200, dtype=torch.complex64) * (1 + 1j)
    out = drop(x)
    dropped = out == 0
    # Real and imaginary parts are dropped together.
    assert torch.equal(out.real == 0, out.imag == 0)
    assert dropped.float().mean().item() == pytest.approx(0.25, abs=0.01)
    torch.testing.assert_close(out[~dropped], x[~dropped] / 0.75)


def test_dropout_rejects_invalid_probability():
    with pytest.raises(ValueError):
        ComplexDropout(1.0)


def test_split_layer_norm_normalises_each_part():
    norm = SplitLayerNorm(32)
    out = norm(randc(4, 10, 32) * 5 + 3)
    for part in (out.real, out.imag):
        torch.testing.assert_close(part.mean(-1), torch.zeros(4, 10), atol=1e-5, rtol=0)
        variance = part.var(-1, unbiased=False)
        torch.testing.assert_close(variance, torch.ones(4, 10), atol=1e-3, rtol=0)


def test_complex_layer_norm_whitens_correlated_input():
    torch.manual_seed(0)
    re = torch.randn(3, 5, 256)
    # Strongly correlated, unequal-variance, off-centre real and imaginary parts.
    im = 0.8 * re + 0.3 * torch.randn(3, 5, 256)
    x = torch.complex(4 * re + 2, im - 1)

    out = ComplexLayerNorm(256, elementwise_affine=False)(x)
    o_re, o_im = out.real, out.imag

    zeros, ones = torch.zeros(3, 5), torch.ones(3, 5)
    torch.testing.assert_close(o_re.mean(-1), zeros, atol=1e-5, rtol=0)
    torch.testing.assert_close(o_im.mean(-1), zeros, atol=1e-5, rtol=0)
    torch.testing.assert_close((o_re * o_re).mean(-1), ones, atol=1e-3, rtol=0)
    torch.testing.assert_close((o_im * o_im).mean(-1), ones, atol=1e-3, rtol=0)
    torch.testing.assert_close((o_re * o_im).mean(-1), zeros, atol=1e-3, rtol=0)


def test_complex_layer_norm_initial_affine_gives_unit_complex_variance():
    out = ComplexLayerNorm(128)(randc(2, 7, 128) * 3)
    power = (out.abs() ** 2).mean(-1)
    torch.testing.assert_close(power, torch.ones(2, 7), atol=1e-3, rtol=0)


def test_complex_layer_norm_handles_purely_real_input():
    # Zero imaginary part makes V singular without eps; the output must stay finite.
    x = torch.complex(torch.randn(2, 16), torch.zeros(2, 16))
    assert torch.isfinite(torch.view_as_real(ComplexLayerNorm(16)(x))).all()


@pytest.mark.parametrize(
    ("layer", "fan_in"),
    [
        (nn.Linear(512, 256, dtype=torch.complex64), 512),
        (nn.Conv2d(16, 64, kernel_size=3, dtype=torch.complex64), 16 * 3 * 3),
    ],
)
def test_complex_kaiming_matches_he_variance(layer, fan_in):
    torch.manual_seed(0)
    complex_kaiming_(layer.weight)
    power = (layer.weight.abs() ** 2).mean().item()
    assert power == pytest.approx(2 / fan_in, rel=0.05)
    phase = layer.weight.angle()
    assert phase.min().item() >= -math.pi
    assert phase.max().item() <= math.pi
    assert phase.mean().item() == pytest.approx(0.0, abs=0.05)


def test_complex_kaiming_rejects_real_tensors():
    with pytest.raises(TypeError):
        complex_kaiming_(torch.empty(3, 3))


def test_init_complex_weights_only_touches_complex_weights():
    model = nn.Sequential(
        nn.Linear(8, 8, dtype=torch.complex64),
        nn.Linear(8, 8),
        nn.Conv2d(3, 4, 2, dtype=torch.complex64),
    )
    real_before = model[1].weight.detach().clone()
    assert init_complex_weights(model) == 2
    assert torch.equal(model[1].weight, real_before)
