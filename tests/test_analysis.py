import json

import torch

from cvit.analysis.curves import load_metrics, plot_curves
from cvit.analysis.rollout import attention_rollout, plot_rollouts, rollout_maps
from cvit.model import ViT, ViTConfig

SMALL = {"embed_dim": 16, "num_heads": 2, "num_blocks": 2, "ff_dim": 32, "image_size": 16}


def random_attention(layers: int, batch: int, heads: int, seq: int) -> list[torch.Tensor]:
    return [torch.rand(batch, heads, seq, seq).softmax(-1) for _ in range(layers)]


def test_rollout_shape_and_mass():
    maps = attention_rollout(random_attention(layers=4, batch=3, heads=2, seq=17))
    assert maps.shape == (3, 4, 4)
    # Row 0 of a product of row-stochastic matrices sums to 1; dropping the class-token column
    # leaves at most 1.
    assert (maps >= 0).all()
    assert (maps.sum(dim=(1, 2)) <= 1 + 1e-6).all()


def test_rollout_multiplies_later_layers_on_the_left():
    torch.manual_seed(0)
    attention = random_attention(layers=2, batch=1, heads=1, seq=5)
    first, second = (0.5 * a[0, 0] + 0.5 * torch.eye(5) for a in attention)
    expected = (second @ first)[0, 1:].reshape(2, 2)
    torch.testing.assert_close(attention_rollout(attention)[0], expected)
    # The opposite order gives a different answer, so the test can tell them apart.
    assert not torch.allclose((first @ second)[0, 1:].reshape(2, 2), expected)


def test_rollout_of_no_attention_is_zero_on_patches():
    # If every token only attends to itself, the class token never reads any patch.
    attention = [torch.eye(5).expand(1, 2, 5, 5) for _ in range(3)]
    torch.testing.assert_close(attention_rollout(attention), torch.zeros(1, 2, 2))


def test_rollout_maps_and_plot_for_both_model_types(tmp_path):
    models = {
        "ViT": ViT(ViTConfig(**SMALL)),
        "CViT": ViT(ViTConfig(complex_valued=True, scoring="linear", **SMALL)),
    }
    images = torch.randn(2, 3, 16, 16)
    for model in models.values():
        preds, maps = rollout_maps(model, images)
        assert preds.shape == (2,)
        assert maps.shape == (2, 16, 16)
        assert maps.min() >= 0
        assert maps.max() <= 1

    out = tmp_path / "rollout.png"
    plot_rollouts(models, images, torch.tensor([0, 1]), out)
    assert out.stat().st_size > 0


def test_plot_curves(tmp_path):
    history = [
        {"epoch": e, "train_loss": 1 / e, "train_acc": 0.1 * e, "test_acc": 0.1 * e}
        for e in (1, 2, 3)
    ]
    run = tmp_path / "run"
    run.mkdir()
    (run / "metrics.json").write_text(json.dumps(history))

    out = tmp_path / "curves.png"
    plot_curves({"run": load_metrics(run)}, out)
    assert out.stat().st_size > 0
