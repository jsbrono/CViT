"""Train a ViT or CViT on CIFAR-10.

Usage::

    python -m cvit.train --config configs/cvit.yaml
    python -m cvit.train --config configs/vit.yaml --seed 1 --epochs 5 --no-augment

Each run writes to ``runs/<config>-seed<seed>/`` (or ``--out``): the resolved config, per-epoch
metrics in ``metrics.json`` (updated every epoch) and the final weights in ``model.pt``.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from cvit.data import cifar10_loaders
from cvit.model import ViT, ViTConfig, count_parameters, save_checkpoint


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 0.0
    augment: bool = True
    num_workers: int = 4
    seed: int = 0


def load_config(path: str | Path) -> tuple[ViTConfig, TrainConfig]:
    """Reads a YAML file with ``model`` and ``train`` sections (both optional)."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    unknown = set(raw) - {"model", "train"}
    if unknown:
        raise ValueError(f"unknown config sections: {sorted(unknown)}")
    return ViTConfig(**raw.get("model", {})), TrainConfig(**raw.get("train", {}))


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    """One pass over ``loader``: trains if an optimizer is given, otherwise evaluates.

    Returns the mean loss and the accuracy.
    """
    training = optimizer is not None
    model.train(training)
    total_loss, correct, seen = 0.0, 0, 0

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, labels)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            seen += len(labels)

    return total_loss / seen, correct / seen


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    config: TrainConfig,
    device: torch.device,
    out_dir: Path | None = None,
) -> list[dict[str, float]]:
    """Trains ``model`` with Adam and evaluates it after every epoch.

    Returns one dict of metrics per epoch. If ``out_dir`` is given, the metrics are written to
    ``out_dir/metrics.json`` after each epoch.
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    history: list[dict[str, float]] = []

    for epoch in range(1, config.epochs + 1):
        start = time.perf_counter()
        train_loss, train_acc = run_epoch(model, train_loader, device, optimizer)
        test_loss, test_acc = run_epoch(model, test_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "seconds": time.perf_counter() - start,
            }
        )
        print(
            f"epoch {epoch:3d}/{config.epochs}  "
            f"train loss {train_loss:.4f} acc {train_acc:.2%}  "
            f"test loss {test_loss:.4f} acc {test_acc:.2%}  "
            f"({history[-1]['seconds']:.0f}s)"
        )
        if out_dir is not None:
            (out_dir / "metrics.json").write_text(json.dumps(history, indent=2))

    return history


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path, help="YAML config file")
    parser.add_argument("--seed", type=int, help="overrides train.seed")
    parser.add_argument("--epochs", type=int, help="overrides train.epochs")
    parser.add_argument("--no-augment", action="store_true", help="disable data augmentation")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, help="output directory (default: runs/<name>)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    model_config, train_config = load_config(args.config)
    overrides = {"seed": args.seed, "epochs": args.epochs}
    train_config = replace(train_config, **{k: v for k, v in overrides.items() if v is not None})
    if args.no_augment:
        train_config = replace(train_config, augment=False)

    out_dir = args.out or Path("runs") / f"{args.config.stem}-seed{train_config.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    resolved = {"model": asdict(model_config), "train": asdict(train_config)}
    (out_dir / "config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False))

    # Model init, augmentation and shuffling all draw from torch's RNG (CPU and CUDA).
    torch.manual_seed(train_config.seed)
    model = ViT(model_config)
    params = count_parameters(model)
    print(
        f"{args.config.stem}: {params['total']:,} parameters "
        f"({params['real_equivalent']:,} real-equivalent), writing to {out_dir}"
    )

    train_loader, test_loader = cifar10_loaders(
        args.data_dir, train_config.batch_size, train_config.num_workers, train_config.augment
    )
    fit(model, train_loader, test_loader, train_config, torch.device(args.device), out_dir)
    save_checkpoint(model, out_dir / "model.pt")
    print(f"saved {out_dir / 'model.pt'}")


if __name__ == "__main__":
    main()
