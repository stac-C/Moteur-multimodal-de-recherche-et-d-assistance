"""Fusion tardive (section 1.3) et évaluation texte -> item en « leave-one-out ».

Un item i (image ou audio) a une modalité propre z_i^m et une moyenne de ses K légendes.
Score tardif : s(q, i) = q . (w_m z_i^m + w_cap c_i) = q . zbar_i  (un seul vecteur fusionné par item).
Leave-one-out : pour une requête qui EST l'une des légendes de l'item i, cette légende est retirée
du contexte de i (sinon la réponse serait dans l'item et les scores seraient gonflés).
"""

from __future__ import annotations

from typing import Callable, Mapping

import numpy as np

from mmsearch.encoders.base import l2_normalize
from mmsearch.eval.metrics import ranks_from_scores, summary


def late_fusion(embs: Mapping[str, np.ndarray], weights: Mapping[str, float]) -> np.ndarray:
    """zbar = somme_m w_m z^m ; w_m >= 0, somme = 1 (vérifié). Retourne float32 (n, d), norme <= 1."""
    if set(embs) != set(weights):
        raise ValueError(f"modalités {sorted(embs)} != poids {sorted(weights)}")
    if any(w < 0 for w in weights.values()) or abs(sum(weights.values()) - 1.0) > 1e-6:
        raise ValueError("poids : w_m >= 0 et somme = 1 requis")
    shapes = {a.shape for a in embs.values()}
    if len(shapes) != 1:
        raise ValueError(f"formes différentes : {shapes}")
    out = sum(float(weights[m]) * embs[m].astype(np.float64) for m in embs)
    return np.ascontiguousarray(out, dtype=np.float32)


def caption_context(cap: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """cap : (n, K, d) normalisés. Retourne (full (n, d), loo (n, K, d)), tous normalisés.

    full[i]   = norm(somme_k cap[i, k])
    loo[i, r] = norm(somme_{k != r} cap[i, k])   (légende r retirée)
    """
    total = cap.sum(axis=1)  # (n, d)
    full = l2_normalize(total)
    loo = l2_normalize(total[:, None, :] - cap)  # (n, K, d)
    return full, loo


def text_to_item_scores_loo(
    q: np.ndarray, z_full: np.ndarray, z_loo: np.ndarray, owner: np.ndarray
) -> np.ndarray:
    """Scores (m, n) de m requêtes contre n items fusionnés.

    q (m, d) ; z_full (n, d) : item avec toutes ses légendes ; z_loo (m, d) : item propriétaire de
    chaque requête, sans la légende-requête ; owner (m,) : indice de l'item propriétaire.
    """
    s = q @ z_full.T
    s[np.arange(len(q)), owner] = np.einsum("md,md->m", q, z_loo)
    return s


def text_to_item_ranks(
    fuse: Callable[[np.ndarray, np.ndarray], np.ndarray], mod: np.ndarray, cap: np.ndarray
) -> np.ndarray:
    """Rang (1 = meilleur) de l'item propriétaire pour chacune des n*K légendes-requêtes (leave-one-out).

    fuse(z_mod (a, d), ctx (a, d)) -> (a, d) : fusion tardive ou précoce.
    mod (n, d) : modalité propre des n items ; cap (n, K, d) : leurs K légendes (toutes servent de requêtes,
    rangées dans l'ordre item 0 (K légendes), item 1, ...).
    """
    n, K, _ = cap.shape
    ctx_full, ctx_loo = caption_context(cap)
    owner = np.repeat(np.arange(n), K)
    z_full = fuse(mod, ctx_full)
    z_loo = fuse(mod[owner], ctx_loo.reshape(n * K, -1))
    scores = text_to_item_scores_loo(cap.reshape(n * K, -1), z_full, z_loo, owner)
    return ranks_from_scores(scores, owner)


def evaluate_text_to_item(
    fuse: Callable[[np.ndarray, np.ndarray], np.ndarray], mod: np.ndarray, cap: np.ndarray
) -> dict[str, float]:
    """R@1, R@5, R@10, MRR, MedR et rsum = R@1 + R@5 + R@10 de `text_to_item_ranks`."""
    m = summary(text_to_item_ranks(fuse, mod, cap))
    m["rsum"] = m["R@1"] + m["R@5"] + m["R@10"]
    return m


def grid_search_weights(
    mod: np.ndarray, cap: np.ndarray, step: float = 0.1
) -> tuple[float, list[dict]]:
    """Choisit w_mod sur la validation (w_cap = 1 - w_mod) selon rsum. Retourne (w_mod, tableau complet)."""
    rows = []
    for w in np.round(np.arange(0.0, 1.0 + 1e-9, step), 6):
        w = float(w)
        m = evaluate_text_to_item(lambda a, b, w=w: w * a + (1.0 - w) * b, mod, cap)
        rows.append({"w_mod": w, **m})
    return max(rows, key=lambda r: r["rsum"])["w_mod"], rows
