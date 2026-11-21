import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from cvit.data import build_transforms, denormalise
from cvit.model import ViT, ViTConfig, count_parameters
from cvit.train import TrainConfig, fit, load_config

CONFIGS = Path(__file__).parent.parent / "configs"
ALL_CONFIGS = sorted(CONFIGS.rglob("*.yaml"))


@pytest.mark.parametrize("path", ALL_CONFIGS, ids=lambda p: p.stem)
def test_every_config_builds_a_model(path):
    model_config, train_config = load_config(path)
    model = ViT(model_config)
    images = torch.randn(2, 3, model_config.image_size, model_config.image_size)
    assert model.eval()(images).shape == (2, model_config.num_classes)
    assert train_config.epochs > 0


def test_main_configs_are_the_dissertation_models():
    vit, _ = load_config(CONFIGS / "vit.yaml")
    cvit, _ = load_config(CONFIGS / "cvit.yaml")
    assert count_parameters(ViT(vit))["total"] == 5_280_778
    assert count_parameters(ViT(cvit))["total"] == 2_812_188


def test_unknown_config_sections_are_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("modle:\n  embed_dim: 16\n")
    with pytest.raises(ValueError, match="modle"):
        load_config(path)


def test_unknown_config_keys_are_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("model:\n  embed_dims: 16\n")
    with pytest.raises(TypeError, match="embed_dims"):
        load_config(path)


@pytest.mark.parametrize("complex_valued", [False, True])
def test_fit_learns_a_separable_toy_problem(tmp_path, complex_valued):
    torch.manual_seed(0)
    # Class 0 images are dark, class 1 images are bright: trivially learnable.
    labels = torch.arange(64) % 2
    images = torch.randn(64, 3, 8, 8) * 0.1 + (labels.view(-1, 1, 1, 1) * 2 - 1).float()
    loader = DataLoader(TensorDataset(images, labels), batch_size=16, shuffle=True)

    config = ViTConfig(
        complex_valued=complex_valued,
        image_size=8,
        num_classes=2,
        embed_dim=16,
        num_heads=2,
        num_blocks=1,
        ff_dim=32,
    )
    history = fit(
        ViT(config), loader, loader, TrainConfig(epochs=5, lr=1e-3), torch.device("cpu"), tmp_path
    )

    assert [h["epoch"] for h in history] == [1, 2, 3, 4, 5]
    assert history[-1]["train_loss"] < history[0]["train_loss"]
    assert history[-1]["test_acc"] == 1.0
    assert json.loads((tmp_path / "metrics.json").read_text()) == history


def test_transforms_produce_normalised_tensors():
    from PIL import Image

    image = Image.new("RGB", (32, 32), color=(128, 64, 200))
    for augment in (False, True):
        train_tf, test_tf = build_transforms(augment)
        assert train_tf(image).shape == (3, 32, 32)
        restored = denormalise(test_tf(image))
        expected = torch.tensor([128, 64, 200]).view(3, 1, 1) / 255
        torch.testing.assert_close(restored, expected.expand(3, 32, 32), atol=1e-6, rtol=0)
