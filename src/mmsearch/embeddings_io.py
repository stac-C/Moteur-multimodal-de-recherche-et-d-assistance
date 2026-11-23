"""Lecture / écriture des embeddings avec vérification du contrat (section 2.7.1).

Un fichier `nom.npy` est toujours accompagné de `nom_ids.json` (ordre des lignes).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def ids_path(path: str | Path) -> Path:
    p = Path(path)
    return p.with_name(p.stem + "_ids.json")


def check_embeddings(
    arr: np.ndarray,
    ids: list[str],
    dim: int = 512,
    normalized: bool = True,
    max_norm: bool = False,
    tol: float = 1e-4,
) -> None:
    """Lève ValueError si le contrat n'est pas respecté.

    normalized=True : toutes les lignes ont une norme L2 = 1 (à tol près).
    max_norm=True   : norme <= 1 + tol (vecteurs fusionnés, section 1.3).
    """
    if arr.ndim != 2 or arr.shape[1] != dim:
        raise ValueError(f"forme attendue (n, {dim}), reçue {arr.shape}")
    if arr.dtype != np.float32:
        raise ValueError(f"dtype attendu float32, reçu {arr.dtype}")
    if not arr.flags["C_CONTIGUOUS"]:
        raise ValueError("tableau non C-contigu")
    if len(ids) != arr.shape[0]:
        raise ValueError(f"{len(ids)} ids pour {arr.shape[0]} lignes")
    if len(set(ids)) != len(ids):
        raise ValueError("ids non uniques")
    if not np.isfinite(arr).all():
        raise ValueError("valeurs non finies (NaN ou inf)")
    norms = np.linalg.norm(arr, axis=1)
    if normalized and np.abs(norms - 1.0).max() > tol:
        raise ValueError(f"normes hors [1 ± {tol}] : min {norms.min():.6f}, max {norms.max():.6f}")
    if max_norm and norms.max() > 1.0 + tol:
        raise ValueError(f"norme > 1 : {norms.max():.6f}")


def save_embeddings(
    path: str | Path,
    arr: np.ndarray,
    ids: list[str],
    dim: int = 512,
    normalized: bool = True,
    max_norm: bool = False,
) -> None:
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    check_embeddings(
        arr, list(ids), dim=dim, normalized=normalized and not max_norm, max_norm=max_norm
    )
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, arr, allow_pickle=False)
    with open(ids_path(p), "w", encoding="utf-8") as f:
        json.dump(list(ids), f, ensure_ascii=False)


def load_embeddings(
    path: str | Path, dim: int = 512, normalized: bool = True, max_norm: bool = False
) -> tuple[np.ndarray, list[str]]:
    p = Path(path)
    arr = np.load(p, allow_pickle=False)
    with open(ids_path(p), encoding="utf-8") as f:
        ids = json.load(f)
    check_embeddings(arr, ids, dim=dim, normalized=normalized and not max_norm, max_norm=max_norm)
    return arr, ids
