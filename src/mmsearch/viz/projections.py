"""Projections 2D (PCA, t-SNE, UMAP) d'un échantillon joint images + légendes + audio (section 2.2, étape 8).

Fichier produit : artifacts/projections/<methode>_joint.json
  {"method", "n_points", "explained_variance" (PCA), "points": [{id, x, y, modality, dataset, label}]}
modality vaut image, caption ou audio ; label = classe ESC-50 pour les clips ESC-50, sinon vide.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np

from mmsearch.config import Config, load_config, set_seed
from mmsearch.data.loaders import load_manifest
from mmsearch.embeddings_io import load_embeddings

METHODS = ("pca", "tsne", "umap")


def build_joint(
    cfg: Config, per_group: int = 1000, seed: int = 42
) -> tuple[np.ndarray, list[dict]]:
    """Échantillonne au plus `per_group` points par groupe (images, légendes, audio Clotho, ESC-50)."""
    rng = np.random.default_rng(seed)
    labels = load_manifest(cfg).set_index("item_id")["label"].to_dict()
    groups = [
        ("clip_img_flickr8k.npy", "image", "flickr8k"),
        ("clip_txt_flickr8k.npy", "caption", "flickr8k"),
        ("audio_clotho.npy", "audio", "clotho"),
        ("audio_esc50.npy", "audio", "esc50"),
    ]
    X, meta = [], []
    for fname, modality, dataset in groups:
        path = cfg.emb_dir / fname
        if not path.exists():
            continue
        emb, ids = load_embeddings(path)
        sel = np.sort(rng.choice(len(ids), size=min(per_group, len(ids)), replace=False))
        X.append(emb[sel])
        meta += [
            {
                "id": ids[i],
                "modality": modality,
                "dataset": dataset,
                "label": labels.get(ids[i], "") if dataset == "esc50" else "",
            }
            for i in sel
        ]
    if not X:
        raise FileNotFoundError(
            "aucun embedding trouvé : lancer l'extraction ou make_fake_artifacts"
        )
    return np.concatenate(X), meta


def pca_2d(X: np.ndarray) -> tuple[np.ndarray, list[float]]:
    """PCA par SVD : coordonnées sur les 2 premiers axes et part de variance expliquée."""
    Xc = X - X.mean(axis=0, keepdims=True)
    _, s, vt = np.linalg.svd(Xc, full_matrices=False)
    var = s**2 / (len(X) - 1)
    return Xc @ vt[:2].T, (var[:2] / var.sum()).tolist()


def project(X: np.ndarray, method: str, seed: int = 42) -> tuple[np.ndarray, Optional[list[float]]]:
    if method == "pca":
        return pca_2d(X)
    if method == "tsne":
        from sklearn.manifold import TSNE

        perplexity = float(min(30, max(5, (len(X) - 1) // 3)))
        return TSNE(
            n_components=2, perplexity=perplexity, init="pca", metric="cosine", random_state=seed
        ).fit_transform(X), None
    if method == "umap":
        import umap

        return umap.UMAP(n_components=2, metric="cosine", random_state=seed).fit_transform(X), None
    raise ValueError(f"méthode inconnue : {method}")


def proj_path(cfg: Config, method: str) -> Path:
    return cfg.artifacts / "projections" / f"{method}_joint.json"


def build_projection(cfg: Config, method: str, per_group: int = 1000) -> Path:
    set_seed(cfg.seed)
    X, meta = build_joint(cfg, per_group, cfg.seed)
    xy, ev = project(X, method, cfg.seed)
    pts = [{**m, "x": round(float(a), 4), "y": round(float(b), 4)} for m, (a, b) in zip(meta, xy)]
    out = proj_path(cfg, method)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"method": method, "n_points": len(pts), "explained_variance": ev, "points": pts}
        ),
        encoding="utf-8",
    )
    return out


def load_or_compute(cfg: Config, method: str) -> dict:
    """Lit la projection précalculée ; la PCA (rapide) est calculée à la demande si absente."""
    if method not in METHODS:
        raise ValueError(f"méthode inconnue : {method}")
    path = proj_path(cfg, method)
    if not path.exists():
        if method != "pca":
            raise FileNotFoundError(
                f"{path.name} absent : lancer python -m mmsearch.viz.projections"
            )
        build_projection(cfg, "pca")
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None)
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    ap.add_argument("--per-group", type=int, default=1000)
    a = ap.parse_args()
    c = load_config(a.config)
    for m in a.methods:
        print("écrit", build_projection(c, m, a.per_group))
