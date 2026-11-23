"""Métriques de recherche : Recall@K, Precision@K, MRR, rang médian (équation E8).

Deux familles d'API :
 - listes (contrat 2.7.3) : `recall_at_k`, `precision_at_k`, `mrr` ;
 - matrices de scores (rapide, utilisé pour l'évaluation) : `ranks_from_scores`,
   `first_relevant_rank`, `recall_from_ranks`, `mrr_from_ranks`.
Le rang d'un item vaut 1 + nombre d'items strictement mieux classés (égalités favorables).
"""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np


def ranks_from_scores(scores: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Rang (1 = meilleur) de l'unique cible de chaque requête. scores : (m, n), target : (m,)."""
    s_true = scores[np.arange(scores.shape[0]), target][:, None]
    return 1 + (scores > s_true).sum(axis=1)


def first_relevant_rank(scores: np.ndarray, relevant: np.ndarray) -> np.ndarray:
    """Rang du premier item pertinent. relevant : masque booléen (m, n), au moins un True par ligne."""
    best = np.where(relevant, scores, -np.inf).max(axis=1)[:, None]
    return 1 + (scores > best).sum(axis=1)


def precision_from_scores(scores: np.ndarray, relevant: np.ndarray, k: int) -> float:
    """Precision@K moyenne : part d'items pertinents parmi les K premiers (relevant : masque (m, n))."""
    top = np.argsort(-scores, axis=1)[:, :k]
    return float(np.take_along_axis(relevant, top, axis=1).mean())


def recall_from_ranks(ranks: np.ndarray, k: int) -> float:
    return float((ranks <= k).mean())


def mrr_from_ranks(ranks: np.ndarray) -> float:
    return float((1.0 / ranks).mean())


def median_rank(ranks: np.ndarray) -> float:
    return float(np.median(ranks))


def summary(ranks: np.ndarray, ks: Sequence[int] = (1, 5, 10)) -> dict[str, float]:
    out = {f"R@{k}": recall_from_ranks(ranks, k) for k in ks}
    out["MRR"] = mrr_from_ranks(ranks)
    out["MedR"] = median_rank(ranks)
    return out


def recall_at_k(
    ranked: Sequence[Sequence[str]],
    relevant: Sequence[set[str]],
    k: int,
    mode: Literal["hit", "fraction"] = "hit",
) -> float:
    vals = []
    for r, rel in zip(ranked, relevant):
        inter = len(set(r[:k]) & rel)
        vals.append(float(inter > 0) if mode == "hit" else inter / len(rel))
    return float(np.mean(vals))


def precision_at_k(ranked: Sequence[Sequence[str]], relevant: Sequence[set[str]], k: int) -> float:
    return float(np.mean([len(set(r[:k]) & rel) / k for r, rel in zip(ranked, relevant)]))


def mrr(ranked: Sequence[Sequence[str]], relevant: Sequence[set[str]]) -> float:
    vals = []
    for r, rel in zip(ranked, relevant):
        pos = next((i + 1 for i, x in enumerate(r) if x in rel), None)
        vals.append(0.0 if pos is None else 1.0 / pos)
    return float(np.mean(vals))
