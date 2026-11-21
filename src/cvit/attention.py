"""Multi-head self-attention for real- and complex-valued tokens."""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import nn

from cvit.complex_nn import ComplexDropout

Scoring = Literal["real", "magnitude", "linear"]
SCORINGS: tuple[str, ...] = ("real", "magnitude", "linear")


class MultiHeadSelfAttention(nn.Module):
    """Scaled dot-product multi-head self-attention.

    For complex tokens the similarity between a query and a key is the Hermitian inner product
    ``q . conj(k)``, which is complex. Softmax needs real logits, so ``scoring`` chooses how to
    get them:

    - ``"real"``: the real part, ``Re(q . conj(k))`` (as in Eilers et al.).
    - ``"magnitude"``: the modulus, ``|q . conj(k)|`` (as in Kothapally et al.).
    - ``"linear"``: a learned real linear layer maps each query's row of scores,
      ``[Re s_1..s_n, Im s_1..s_n]``, to ``n`` logits. This is the method proposed in the
      dissertation. The layer is shared across heads and its size depends on the sequence
      length, so the model is tied to one image size.

    Real-valued attention is the standard transformer attention, and only accepts ``"real"``.

    ``forward`` returns the output tokens and the attention weights (after softmax, always
    real, shape ``(batch, heads, seq, seq)``).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        seq_len: int,
        complex_valued: bool,
        scoring: Scoring = "real",
        dropout: float = 0.1,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )
        if scoring not in SCORINGS:
            raise ValueError(f"unknown scoring {scoring!r}; expected one of {SCORINGS}")
        if not complex_valued and scoring != "real":
            raise ValueError("real-valued attention only supports scoring='real'")

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.complex_valued = complex_valued
        self.scoring = scoring

        dtype = torch.complex64 if complex_valued else None
        self.query = nn.Linear(embed_dim, embed_dim, bias=False, dtype=dtype)
        self.key = nn.Linear(embed_dim, embed_dim, bias=False, dtype=dtype)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False, dtype=dtype)
        self.dropout = ComplexDropout(dropout) if complex_valued else nn.Dropout(dropout)

        if scoring == "linear":
            self.score_proj = nn.Linear(2 * seq_len, seq_len)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        return x.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _logits(self, scores: torch.Tensor) -> torch.Tensor:
        if not self.complex_valued:
            return scores
        if self.scoring == "real":
            return scores.real
        if self.scoring == "magnitude":
            return scores.abs()
        return self.score_proj(torch.cat([scores.real, scores.imag], dim=-1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, embed_dim = x.shape
        q = self._split_heads(self.query(x))
        k = self._split_heads(self.key(x))
        v = self._split_heads(self.value(x))

        # conj() is a no-op for real tensors, so this is q @ k^T in the real case.
        scores = (q @ k.transpose(-2, -1).conj()) * self.scale
        weights = self._logits(scores).softmax(dim=-1)

        out = weights.to(v.dtype) @ v
        out = out.transpose(1, 2).reshape(batch, seq_len, embed_dim)
        return self.dropout(self.out_proj(out)), weights
