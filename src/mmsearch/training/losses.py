"""Perte contrastive InfoNCE symétrique (équation E5)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

LOGIT_SCALE_MAX = math.log(100.0)


def _symmetric_ce(logits: torch.Tensor) -> torch.Tensor:
    """logits[i, j] = S_ij = z_i^x . z_j^y / tau ; positifs sur la diagonale.

    L_{x->y} : softmax par ligne ; L_{y->x} : softmax par colonne (= lignes de logits.T).
    """
    labels = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def info_nce(zx: torch.Tensor, zy: torch.Tensor, logit_scale: torch.Tensor) -> torch.Tensor:
    """zx, zy : (N, d) normalisés ; logit_scale = ln(1/tau) (paramètre appris, borné à ln 100)."""
    scale = logit_scale.clamp(max=LOGIT_SCALE_MAX).exp()
    return _symmetric_ce(scale * zx @ zy.T)


def info_nce_tau(zx: torch.Tensor, zy: torch.Tensor, tau: float) -> torch.Tensor:
    """Variante à température fixe (tests numériques)."""
    return _symmetric_ce(zx @ zy.T / tau)
