"""Contrat commun des encodeurs (section 2.7.3)."""

from __future__ import annotations

from typing import Any, Iterator, Protocol, Sequence

import numpy as np

EMB_DIM = 512


class Encoder(Protocol):
    modality: str  # "text" | "image" | "audio"

    def encode(self, inputs: Sequence[Any], batch_size: int = 64) -> np.ndarray:
        """Retourne float32 (n, 512), lignes de norme L2 = 1."""
        ...


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    """Normalisation L2 : z = u / max(||u||, eps)."""
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return (x / np.maximum(norm, eps)).astype(np.float32)


def batched(seq: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]
