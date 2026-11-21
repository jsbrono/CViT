"""CIFAR-10 data loading and augmentation."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR10_CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck",
)  # fmt: skip


def build_transforms(augment: bool) -> tuple[transforms.Compose, transforms.Compose]:
    """Returns ``(train_transform, test_transform)``.

    The augmentations are the strong set from the dissertation, which the complex model needed
    to stop it overfitting: random crops, flips, colour jitter and random erasing.
    """
    normalise = [transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)]
    test_transform = transforms.Compose(normalise)
    if not augment:
        return transforms.Compose(normalise), test_transform

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            *normalise,
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.2)),
        ]
    )
    return train_transform, test_transform


def cifar10_loaders(
    root: str | Path, batch_size: int, num_workers: int, augment: bool
) -> tuple[DataLoader, DataLoader]:
    """Returns ``(train_loader, test_loader)`` for CIFAR-10, downloading it to ``root``."""
    train_transform, test_transform = build_transforms(augment)
    train_set = datasets.CIFAR10(root, train=True, transform=train_transform, download=True)
    test_set = datasets.CIFAR10(root, train=False, transform=test_transform, download=True)

    loader_args = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    return (
        DataLoader(train_set, shuffle=True, drop_last=True, **loader_args),
        DataLoader(test_set, shuffle=False, **loader_args),
    )


def denormalise(images: torch.Tensor) -> torch.Tensor:
    """Undoes the normalisation of a ``(..., 3, H, W)`` batch, clamped to ``[0, 1]``."""
    mean = torch.tensor(CIFAR10_MEAN, device=images.device).view(3, 1, 1)
    std = torch.tensor(CIFAR10_STD, device=images.device).view(3, 1, 1)
    return (images * std + mean).clamp(0, 1)
