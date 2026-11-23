"""Chargement du manifest, des légendes et alignement des légendes sur les items."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mmsearch.config import Config


def load_manifest(cfg: Config) -> pd.DataFrame:
    return pd.read_csv(cfg.processed / "manifest.csv", dtype=str, keep_default_na=False)


def load_captions(cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(
        cfg.processed / "captions.csv",
        dtype={"caption_id": str, "item_id": str, "text": str},
        keep_default_na=False,
    )
    df["idx"] = df["idx"].astype(int)
    return df


def captions_tensor(
    item_ids: list[str],
    captions: pd.DataFrame,
    cap_emb: np.ndarray,
    cap_ids: list[str],
    n_caps: int = 5,
) -> np.ndarray:
    """Retourne un tenseur (n_items, n_caps, d) : les embeddings des légendes de chaque item.

    L'ordre des items est celui de `item_ids` ; les légendes sont triées par `idx`.
    Lève ValueError si un item a moins de `n_caps` légendes.
    """
    row_of = {cid: i for i, cid in enumerate(cap_ids)}
    by_item = {
        k: g.sort_values("idx")["caption_id"].tolist() for k, g in captions.groupby("item_id")
    }
    out = np.zeros((len(item_ids), n_caps, cap_emb.shape[1]), dtype=np.float32)
    for i, it in enumerate(item_ids):
        cids = by_item.get(it, [])
        if len(cids) < n_caps:
            raise ValueError(f"{it} : {len(cids)} légendes, {n_caps} attendues")
        out[i] = cap_emb[[row_of[c] for c in cids[:n_caps]]]
    return out
