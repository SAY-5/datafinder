"""Embedder protocol + a deterministic in-process implementation.

The deterministic embedder hashes each input string into a fixed-
dimension float vector. Identical inputs produce identical vectors;
inputs that share tokens land closer in cosine distance than inputs
that don't. That's all we need for tests — there's no learning,
just enough structure to exercise the search code.
"""

from __future__ import annotations

import hashlib
import math
from typing import Protocol


class Embedder(Protocol):
    """Anything that can turn texts into fixed-dim vectors."""

    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class DeterministicEmbedder:
    """Token-bag hash embedder. Pure Python, no deps.

    Each input is lowercased + split on non-alphanumerics. Each token
    contributes to a slice of the output vector via a stable hash.
    The vector is L2-normalized so cosine similarity reduces to a
    dot product. Two inputs that share tokens end up correlated.
    """

    def __init__(self, dim: int = 256) -> None:
        if dim < 16:
            raise ValueError("dim too small to be useful")
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    # ----- internals -------------------------------------------------

    def _embed_one(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for tok in self._tokens(text):
            h = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
            i = int.from_bytes(h[:4], "little") % self.dim
            sign = 1.0 if (h[4] & 1) else -1.0
            v[i] += sign
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    @staticmethod
    def _tokens(s: str) -> list[str]:
        out: list[str] = []
        cur: list[str] = []
        for c in s.lower():
            if c.isalnum():
                cur.append(c)
            elif cur:
                out.append("".join(cur))
                cur.clear()
        if cur:
            out.append("".join(cur))
        return [t for t in out if len(t) > 1]


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity. Inputs assumed already L2-normalized."""
    return sum(x * y for x, y in zip(a, b, strict=True))
