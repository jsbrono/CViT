import itertools

import pytest
import torch
from torch import nn

from cvit.attention import MultiHeadSelfAttention
from cvit.model import ViT, ViTConfig, count_parameters, load_checkpoint, save_checkpoint

# The final models compared in the dissertation (Appendix C).
DISSERTATION_VIT = ViTConfig(embed_dim=256, ff_dim=1024, activation="gelu")
DISSERTATION_CVIT = ViTConfig(
    complex_valued=True, embed_dim=176, ff_dim=704, scoring="linear", activation="relu"
)

SMALL = {"embed_dim": 16, "num_heads": 2, "num_blocks": 2, "ff_dim": 32, "image_size": 16}

COMPLEX_VARIANTS = [
    ViTConfig(complex_valued=True, patch_stem=stem, scoring=scoring, layer_norm=norm, **SMALL)
    for stem, scoring, norm in itertools.product(
        ("linear", "conv"), ("real", "magnitude", "linear"), ("whitening", "split")
    )
]
REAL_VARIANTS = [ViTConfig(patch_stem=stem, **SMALL) for stem in ("linear", "conv")]


def test_parameter_counts_match_dissertation_table_4_6():
    vit = count_parameters(ViT(DISSERTATION_VIT))
    cvit = count_parameters(ViT(DISSERTATION_CVIT))
    assert vit["total"] == vit["real_equivalent"] == 5_280_778
    assert cvit["total"] == 2_812_188
    assert cvit["real_equivalent"] == 5_049_852


@pytest.mark.parametrize("config", REAL_VARIANTS + COMPLEX_VARIANTS, ids=repr)
def test_forward_shapes_and_attention(config):
    model = ViT(config).eval()
    images = torch.randn(3, 3, config.image_size, config.image_size)
    logits, attention = model(images, return_attention=True)

    assert logits.shape == (3, config.num_classes)
    assert logits.dtype == torch.float32
    assert len(attention) == config.num_blocks
    for weights in attention:
        assert weights.shape == (3, config.num_heads, config.seq_len, config.seq_len)
        assert not weights.is_complex()
        torch.testing.assert_close(weights.sum(-1), torch.ones(weights.shape[:-1]))


@pytest.mark.parametrize("config", REAL_VARIANTS + COMPLEX_VARIANTS, ids=repr)
def test_training_step_updates_every_parameter(config):
    torch.manual_seed(0)
    model = ViT(config).train()
    images = torch.randn(4, 3, config.image_size, config.image_size)
    labels = torch.randint(0, config.num_classes, (4,))

    loss = nn.functional.cross_entropy(model(images), labels)
    loss.backward()

    assert torch.isfinite(loss)
    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(
            torch.view_as_real(param.grad) if param.is_complex() else param.grad
        ).all()


def test_eval_mode_is_deterministic():
    model = ViT(COMPLEX_VARIANTS[0]).eval()
    images = torch.randn(2, 3, 16, 16)
    torch.testing.assert_close(model(images), model(images))


def test_real_attention_matches_pytorch_reference():
    torch.manual_seed(0)
    attn = MultiHeadSelfAttention(32, 4, seq_len=5, complex_valued=False, dropout=0.0)
    x = torch.randn(2, 5, 32)
    out, _ = attn(x)

    def heads(t):
        return t.view(2, 5, 4, 8).transpose(1, 2)

    q, k, v = heads(attn.query(x)), heads(attn.key(x)), heads(attn.value(x))
    expected = nn.functional.scaled_dot_product_attention(q, k, v)
    expected = attn.out_proj(expected.transpose(1, 2).reshape(2, 5, 32))
    torch.testing.assert_close(out, expected)


def test_complex_attention_uses_hermitian_product():
    # Identity projections, tokens a = (i, i, i, i) and b = (1, 1, 1, 1), scale 1/sqrt(4).
    # Hermitian: Re<a, a> = |a|^2 = 4 and Re<a, b> = Re(4i) = 0, so a's logits are (2, 0).
    # Without the conjugate, <a, a> = 4 i^2 = -4 and a's logits would be (-2, 0) instead.
    attn = MultiHeadSelfAttention(4, 1, seq_len=2, complex_valued=True, scoring="real")
    with torch.no_grad():
        for layer in (attn.query, attn.key):
            layer.weight.copy_(torch.eye(4, dtype=torch.complex64))
    x = torch.tensor([[[1j] * 4, [1] * 4]], dtype=torch.complex64)
    _, weights = attn.eval()(x)

    expected = torch.softmax(torch.tensor([2.0, 0.0]), dim=0)
    torch.testing.assert_close(weights[0, 0, 0], expected)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"image_size": 30, "patch_size": 4},
        {"scoring": "linear"},  # real models only support the plain dot product
        {"patch_stem": "pixels"},
        {"layer_norm": "batch"},
        {"activation": "tanh"},
    ],
)
def test_invalid_configs_are_rejected(kwargs):
    with pytest.raises(ValueError):
        ViTConfig(**kwargs)


def test_checkpoint_round_trip(tmp_path):
    model = ViT(COMPLEX_VARIANTS[-1]).eval()
    path = tmp_path / "nested" / "model.pt"
    save_checkpoint(model, path)
    restored = load_checkpoint(path)

    assert restored.config == model.config
    images = torch.randn(2, 3, 16, 16)
    torch.testing.assert_close(restored(images), model(images))
