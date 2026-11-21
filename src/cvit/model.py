"""Real- and complex-valued Vision Transformers sharing one implementation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
from torch import nn

from cvit.attention import SCORINGS, MultiHeadSelfAttention, Scoring
from cvit.complex_nn import (
    ComplexDropout,
    ComplexLayerNorm,
    SplitGELU,
    SplitLayerNorm,
    SplitReLU,
    init_complex_weights,
)


@dataclass
class ViTConfig:
    """Architecture of a :class:`ViT`.

    ``scoring`` and ``layer_norm`` only apply to complex-valued models (a real model must use
    ``scoring="real"``, the standard dot product). ``activation`` applies to both; complex
    models apply it to the real and imaginary parts separately.
    """

    complex_valued: bool = False
    image_size: int = 32
    patch_size: int = 4
    in_channels: int = 3
    num_classes: int = 10
    embed_dim: int = 256
    num_heads: int = 8
    num_blocks: int = 6
    ff_dim: int = 1024
    patch_stem: Literal["linear", "conv"] = "conv"
    scoring: Scoring = "real"
    layer_norm: Literal["whitening", "split"] = "whitening"
    activation: Literal["relu", "gelu"] = "gelu"
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.image_size % self.patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        if self.patch_stem not in ("linear", "conv"):
            raise ValueError(f"unknown patch_stem {self.patch_stem!r}")
        if self.scoring not in SCORINGS:
            raise ValueError(f"unknown scoring {self.scoring!r}")
        if not self.complex_valued and self.scoring != "real":
            raise ValueError("real-valued models only support scoring='real'")
        if self.layer_norm not in ("whitening", "split"):
            raise ValueError(f"unknown layer_norm {self.layer_norm!r}")
        if self.activation not in ("relu", "gelu"):
            raise ValueError(f"unknown activation {self.activation!r}")
        if self.patch_stem == "conv" and self.embed_dim % 2 != 0:
            raise ValueError("the conv patch stem needs an even embed_dim")

    @property
    def grid_size(self) -> int:
        """Number of patches along each side of the image."""
        return self.image_size // self.patch_size

    @property
    def seq_len(self) -> int:
        """Number of tokens: one per patch, plus the class token."""
        return self.grid_size**2 + 1


class ConvStem(nn.Module):
    """A 3x3 convolution followed by a patchifying convolution, each with BatchNorm and GELU.

    A small convolutional stem in place of the single linear patch projection makes ViTs much
    easier to train on small datasets (Xiao et al., 2021, "Early Convolutions Help Transformers
    See Better").
    """

    def __init__(self, in_channels: int, embed_dim: int, patch_size: int):
        super().__init__()
        hidden = embed_dim // 2
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, embed_dim, kernel_size=patch_size, stride=patch_size),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ComplexConvStem(nn.Module):
    """Two independent real :class:`ConvStem` s produce the real and imaginary parts."""

    def __init__(self, in_channels: int, embed_dim: int, patch_size: int):
        super().__init__()
        self.real = ConvStem(in_channels, embed_dim, patch_size)
        self.imag = ConvStem(in_channels, embed_dim, patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.complex(self.real(x), self.imag(x))


class ComplexLinearStem(nn.Module):
    """Complex linear patch projection. Real images are cast to complex with zero phase."""

    def __init__(self, in_channels: int, embed_dim: int, patch_size: int):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels, embed_dim, patch_size, stride=patch_size, dtype=torch.complex64
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.to(torch.complex64))


def _patch_stem(config: ViTConfig) -> nn.Module:
    args = (config.in_channels, config.embed_dim, config.patch_size)
    if config.patch_stem == "conv":
        return ComplexConvStem(*args) if config.complex_valued else ConvStem(*args)
    if config.complex_valued:
        return ComplexLinearStem(*args)
    return nn.Conv2d(*args, stride=config.patch_size)


class EncoderBlock(nn.Module):
    """Pre-norm transformer encoder block: ``x + Attn(Norm(x))``, then ``x + MLP(Norm(x))``."""

    def __init__(self, config: ViTConfig):
        super().__init__()
        dim, hidden = config.embed_dim, config.ff_dim
        self.attention = MultiHeadSelfAttention(
            dim,
            config.num_heads,
            seq_len=config.seq_len,
            complex_valued=config.complex_valued,
            scoring=config.scoring,
            dropout=config.dropout,
        )

        if config.complex_valued:
            norm = ComplexLayerNorm if config.layer_norm == "whitening" else SplitLayerNorm
            activation = SplitReLU() if config.activation == "relu" else SplitGELU()
            dtype = torch.complex64
            dropout: nn.Module = ComplexDropout(config.dropout)
        else:
            norm = nn.LayerNorm
            activation = nn.ReLU() if config.activation == "relu" else nn.GELU()
            dtype = None
            dropout = nn.Dropout(config.dropout)

        self.norm1 = norm(dim)
        self.norm2 = norm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden, dtype=dtype),
            activation,
            nn.Linear(hidden, dim, dtype=dtype),
            dropout,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attended, weights = self.attention(self.norm1(x))
        x = x + attended
        x = x + self.mlp(self.norm2(x))
        return x, weights


class ViT(nn.Module):
    """Vision Transformer with real- or complex-valued tokens.

    The class token and positional embeddings are real-valued in both variants (as in the
    dissertation); in the complex model they are added to the real part of the patch tokens,
    and the first complex projection mixes them into the imaginary part. The complex model's
    classifier is a real linear layer on ``[Re(cls), Im(cls)]``.
    """

    def __init__(self, config: ViTConfig):
        super().__init__()
        self.config = config
        dim = config.embed_dim

        self.patch_embed = _patch_stem(config)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.randn(1, config.seq_len, dim))
        self.blocks = nn.ModuleList(EncoderBlock(config) for _ in range(config.num_blocks))
        head_in = 2 * dim if config.complex_valued else dim
        self.head = nn.Linear(head_in, config.num_classes)

        if config.complex_valued:
            init_complex_weights(self)

    def forward(
        self, images: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Classifies a batch of images of shape ``(batch, channels, height, width)``.

        Returns logits of shape ``(batch, num_classes)``, and, if ``return_attention``, the
        attention weights of every block, each of shape ``(batch, heads, seq, seq)``.
        """
        tokens = self.patch_embed(images).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(tokens.shape[0], -1, -1).to(tokens.dtype)
        x = torch.cat([cls, tokens], dim=1) + self.pos_embed

        attention = []
        for block in self.blocks:
            x, weights = block(x)
            attention.append(weights)

        cls_out = x[:, 0]
        if self.config.complex_valued:
            cls_out = torch.cat([cls_out.real, cls_out.imag], dim=-1)
        logits = self.head(cls_out)
        return (logits, attention) if return_attention else logits


def count_parameters(model: nn.Module) -> dict[str, float]:
    """Parameter counts and storage size.

    ``total`` counts a complex weight as one parameter; ``real_equivalent`` counts it as two
    (real and imaginary part), which is the fair figure when comparing to a real model.
    """
    params = list(model.parameters())
    return {
        "total": sum(p.numel() for p in params),
        "real_equivalent": sum(p.numel() * (2 if p.is_complex() else 1) for p in params),
        "megabytes": sum(p.numel() * p.element_size() for p in params) / 1024**2,
    }


def save_checkpoint(model: ViT, path: str | Path) -> None:
    """Saves the model's weights together with the config needed to rebuild it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": asdict(model.config), "state_dict": model.state_dict()}, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> ViT:
    """Rebuilds a :class:`ViT` saved with :func:`save_checkpoint`, in eval mode."""
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    model = ViT(ViTConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval()
