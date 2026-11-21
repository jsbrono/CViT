"""Attention rollout (Abnar & Zuidema, 2020, "Quantifying Attention Flow in Transformers").

Shows which image patches the class token draws information from, across all layers::

    python -m cvit.analysis.rollout runs/vit-seed0/model.pt runs/cvit-seed0/model.pt \\
        --num-images 8 --out results/rollout.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from matplotlib.figure import Figure
from torchvision import datasets

from cvit.data import CIFAR10_CLASSES, build_transforms, denormalise
from cvit.model import ViT, load_checkpoint


def attention_rollout(attention: list[torch.Tensor]) -> torch.Tensor:
    """Combines per-layer attention weights into one map of class-token attention to patches.

    Each layer's weights ``(batch, heads, seq, seq)`` are averaged over heads, mixed with the
    identity to account for the residual connection (``0.5 A + 0.5 I``), and multiplied from
    the first layer to the last: ``R = A_L ... A_2 A_1``. Row 0 of ``R`` (the class token),
    without the class token's own column, is the result.

    Returns a ``(batch, grid, grid)`` tensor whose entries sum to at most 1 per image.
    """
    seq_len = attention[0].shape[-1]
    identity = torch.eye(seq_len, device=attention[0].device)
    rollout = identity
    for weights in attention:
        layer = 0.5 * weights.mean(dim=1) + 0.5 * identity
        rollout = layer @ rollout

    grid = int(round((seq_len - 1) ** 0.5))
    return rollout[:, 0, 1:].reshape(-1, grid, grid)


@torch.no_grad()
def rollout_maps(model: ViT, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns the predictions and rollout maps upsampled to image size, scaled to ``[0, 1]``."""
    logits, attention = model.eval()(images, return_attention=True)
    maps = attention_rollout(attention).unsqueeze(1)
    maps = F.interpolate(maps, size=images.shape[-2:], mode="bilinear", align_corners=False)
    maps = maps.squeeze(1)
    low = maps.amin(dim=(1, 2), keepdim=True)
    high = maps.amax(dim=(1, 2), keepdim=True)
    return logits.argmax(dim=1), (maps - low) / (high - low).clamp_min(1e-12)


def plot_rollouts(
    models: dict[str, ViT], images: torch.Tensor, labels: torch.Tensor, out: Path
) -> None:
    """One row per image: the image, then each model's rollout overlaid on it."""
    results = {name: rollout_maps(model, images) for name, model in models.items()}
    pictures = denormalise(images).permute(0, 2, 3, 1).numpy()

    n_rows, n_cols = len(images), len(models) + 1
    fig = Figure(figsize=(2 * n_cols, 2 * n_rows))
    axes = fig.subplots(n_rows, n_cols, squeeze=False)
    for row in range(n_rows):
        axes[row, 0].imshow(pictures[row])
        axes[row, 0].set_title(CIFAR10_CLASSES[labels[row]], fontsize=9)
        for col, (name, (preds, maps)) in enumerate(results.items(), start=1):
            axes[row, col].imshow(pictures[row])
            axes[row, col].imshow(maps[row].numpy(), cmap="inferno", alpha=0.55)
            verdict = "correct" if preds[row] == labels[row] else CIFAR10_CLASSES[preds[row]]
            axes[row, col].set_title(f"{name}: {verdict}", fontsize=9)
    for ax in axes.flat:
        ax.axis("off")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Plot attention rollout for trained models.")
    parser.add_argument("checkpoints", nargs="+", type=Path, help="model.pt files to compare")
    parser.add_argument("--num-images", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0, help="chooses which test images to show")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("results/rollout.png"))
    args = parser.parse_args(argv)

    _, test_transform = build_transforms(augment=False)
    test_set = datasets.CIFAR10(args.data_dir, train=False, transform=test_transform, download=True)
    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(test_set), generator=generator)[: args.num_images].tolist()
    images = torch.stack([test_set[i][0] for i in indices])
    labels = torch.tensor([test_set[i][1] for i in indices])

    models = {}
    for path in args.checkpoints:
        model = load_checkpoint(path)
        models["CViT" if model.config.complex_valued else "ViT"] = model
    if len(models) != len(args.checkpoints):
        # Fall back to directory names when two checkpoints share a model type.
        models = {path.parent.name: load_checkpoint(path) for path in args.checkpoints}

    plot_rollouts(models, images, labels, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
