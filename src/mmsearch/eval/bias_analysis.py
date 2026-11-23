"""Analyse des biais (tâche A8) : statistiques des légendes et performance par sous-groupe de requêtes.

Sortie : artifacts/figures/bias_report.json
 - dataset : longueur des légendes, mots les plus fréquents, part de légendes par groupe thématique ;
 - retrieval : Recall@1 / Recall@10 texte -> image par groupe, pour la fusion tardive et pour l'image seule,
   avec intervalle de confiance à 95 % (approximation normale) et écart de disparité entre groupes.
Lancer : python -m mmsearch.eval.bias_analysis
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter

import numpy as np
import pandas as pd

from mmsearch.config import load_config
from mmsearch.data.loaders import load_captions
from mmsearch.eval.run_eval import late, load_flickr
from mmsearch.fusion.late import text_to_item_ranks

GROUPS = {
    "hommes": {"man", "men", "boy", "boys"},
    "femmes": {"woman", "women", "girl", "girls"},
    "enfants": {"child", "children", "kid", "kids"},
    "animaux": {"dog", "dogs", "cat", "cats", "horse", "horses", "bird", "birds"},
    "nature": {"grass", "snow", "beach", "water", "mountain", "mountains", "field", "ocean"},
    "ville": {"street", "city", "building", "road", "car", "bus"},
}
STOP = set(
    "a an the of in on at with and is are to his her its it their by for from while up down into over "
    "his as two one other another out near some there that".split()
)


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())


def groups_of(text: str) -> list[str]:
    toks = set(tokens(text))
    return [g for g, words in GROUPS.items() if toks & words]


def dataset_stats(captions: pd.DataFrame) -> dict:
    toks = [tokens(t) for t in captions["text"]]
    lens = np.array([len(t) for t in toks])
    counts = Counter(w for t in toks for w in t if w not in STOP)
    n = len(toks)
    share = {g: float(np.mean([bool(set(t) & words) for t in toks])) for g, words in GROUPS.items()}
    m, f = share["hommes"], share["femmes"]
    return {
        "n_captions": n,
        "mean_len": float(lens.mean()),
        "std_len": float(lens.std()),
        "top_words": counts.most_common(20),
        "share_by_group": share,
        "male_to_female_ratio": (m / f) if f else None,
    }


def _ci95(p: float, n: int) -> float:
    return float(1.96 * np.sqrt(p * (1 - p) / n)) if n else 0.0


def group_retrieval(
    query_texts: list[str], ranks_by_system: dict[str, np.ndarray], min_n: int = 30
) -> dict:
    """query_texts : n*K textes de requête ; ranks_by_system[s] : rang de l'item propriétaire (n*K,)."""
    member = {g: np.array([g in groups_of(t) for t in query_texts]) for g in GROUPS}
    member["autres"] = ~np.any(list(member.values()), axis=0)
    out: dict = {"groups": {}, "disparity_R@10": {}}
    for g, mask in member.items():
        n = int(mask.sum())
        row = {"n_queries": n}
        for system, ranks in ranks_by_system.items():
            for k in (1, 10):
                p = float((ranks[mask] <= k).mean()) if n else float("nan")
                row[f"{system} R@{k}"] = p
                if k == 10:
                    row[f"{system} R@10 ci95"] = _ci95(p, n) if n else float("nan")
        out["groups"][g] = row
    for system in ranks_by_system:
        vals = {g: r[f"{system} R@10"] for g, r in out["groups"].items() if r["n_queries"] >= min_n}
        if vals:
            out["disparity_R@10"][system] = {
                "max_group": max(vals, key=vals.get),
                "min_group": min(vals, key=vals.get),
                "gap": float(max(vals.values()) - min(vals.values())),
            }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--w", type=float, default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    w = cfg["fusion"]["weights_image"]["image"] if a.w is None else a.w
    fl = load_flickr(cfg, a.split)
    texts = [t for item in fl["texts"] for t in item]
    ranks = {
        "fusion": text_to_item_ranks(late(w), fl["img"], fl["cap"]),
        "image seule": text_to_item_ranks(lambda x, y: x, fl["img"], fl["cap"]),
    }
    caps = load_captions(cfg)
    caps = caps[caps["item_id"].str.startswith("img_")]
    report = {
        "w_img": w,
        "split": a.split,
        "dataset": dataset_stats(caps),
        "retrieval": group_retrieval(texts, ranks),
    }
    out = cfg.artifacts / "figures" / "bias_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["retrieval"]["disparity_R@10"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
