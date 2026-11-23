"""Artefacts factices (section 2.7.6) : mêmes formes et formats que les vrais, graine fixe.

Les vecteurs sont corrélés (légendes et clips proches de leur vecteur de base) pour que la recherche
donne des résultats non triviaux. Aucune donnée réelle, aucun modèle.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from mmsearch.config import ROOT
from mmsearch.data.ingest import MANIFEST_COLS
from mmsearch.embeddings_io import save_embeddings

CLASSES = ["dog", "rooster", "pig", "cow", "frog", "cat", "hen", "insects", "sheep", "crow"]


def _unit(x: np.ndarray) -> np.ndarray:
    return (x / np.linalg.norm(x, axis=-1, keepdims=True)).astype(np.float32)


def _split(n: int, i: int) -> str:
    return "train" if i < 0.6 * n else ("val" if i < 0.8 * n else "test")


def make_fake_artifacts(
    root: Path, n_img: int = 200, n_aud: int = 100, n_esc: int = 50, seed: int = 42, dim: int = 512
) -> Path:
    rng = np.random.default_rng(seed)
    root = Path(root)
    (root / "data" / "processed").mkdir(parents=True, exist_ok=True)
    if not (root / "config.yaml").exists():
        shutil.copy(ROOT / "config.yaml", root / "config.yaml")
    items, caps = [], []
    for i in range(n_img):
        items.append(
            dict(
                item_id=f"img_{i:06d}",
                modality="image",
                dataset="flickr8k",
                split=_split(n_img, i),
                path=f"data/raw/flickr8k/Images/fake_{i:06d}.jpg",
                label="",
                duration_s="",
            )
        )
        caps += [
            dict(
                caption_id=f"img_{i:06d}_c{j}",
                item_id=f"img_{i:06d}",
                idx=j,
                text=f"fake caption {i} {j}",
            )
            for j in range(5)
        ]
    for i in range(n_aud):
        items.append(
            dict(
                item_id=f"aud_{i:06d}",
                modality="audio",
                dataset="clotho",
                split=_split(n_aud, i),
                path=f"data/raw/clotho/fake_{i:06d}.wav",
                label="",
                duration_s="20.00",
            )
        )
        caps += [
            dict(
                caption_id=f"aud_{i:06d}_c{j}",
                item_id=f"aud_{i:06d}",
                idx=j,
                text=f"fake sound {i} {j}",
            )
            for j in range(5)
        ]
    labels = [CLASSES[i % len(CLASSES)] for i in range(n_esc)]
    for i in range(n_esc):
        items.append(
            dict(
                item_id=f"esc_{i:06d}",
                modality="audio",
                dataset="esc50",
                split="test",
                path=f"data/raw/esc50/audio/fake_{i:06d}.wav",
                label=labels[i],
                duration_s="5.00",
            )
        )
    pd.DataFrame(items, columns=MANIFEST_COLS).to_csv(
        root / "data" / "processed" / "manifest.csv", index=False
    )
    pd.DataFrame(caps, columns=["caption_id", "item_id", "idx", "text"]).to_csv(
        root / "data" / "processed" / "captions.csv", index=False
    )

    E = root / "artifacts" / "embeddings"
    base_i, base_a = _unit(rng.normal(size=(n_img, dim))), _unit(rng.normal(size=(n_aud, dim)))
    img_ids = [f"img_{i:06d}" for i in range(n_img)]
    aud_ids = [f"aud_{i:06d}" for i in range(n_aud)]
    save_embeddings(
        E / "clip_img_flickr8k.npy",
        _unit(base_i + 0.3 * rng.normal(size=base_i.shape) / np.sqrt(dim) * 4),
        img_ids,
    )
    cap_i = _unit(base_i[:, None, :] + 0.6 * rng.normal(size=(n_img, 5, dim)) / np.sqrt(dim) * 4)
    save_embeddings(
        E / "clip_txt_flickr8k.npy",
        cap_i.reshape(-1, dim),
        [f"{i}_c{j}" for i in img_ids for j in range(5)],
    )
    save_embeddings(
        E / "audio_clotho.npy",
        _unit(base_a + 0.5 * rng.normal(size=base_a.shape) / np.sqrt(dim) * 4),
        aud_ids,
    )
    cap_a = _unit(base_a[:, None, :] + 0.6 * rng.normal(size=(n_aud, 5, dim)) / np.sqrt(dim) * 4)
    save_embeddings(
        E / "clip_txt_clotho.npy",
        cap_a.reshape(-1, dim),
        [f"{i}_c{j}" for i in aud_ids for j in range(5)],
    )
    save_embeddings(
        E / "ast_raw_clotho.npy",
        rng.normal(size=(n_aud, 768)).astype(np.float32),
        aud_ids,
        dim=768,
        normalized=False,
    )
    cls_vec = _unit(rng.normal(size=(len(CLASSES), dim)))
    esc = _unit(
        cls_vec[[CLASSES.index(lab) for lab in labels]]
        + 0.4 * rng.normal(size=(n_esc, dim)) / np.sqrt(dim) * 4
    )
    proj = rng.normal(size=(dim, 384)).astype(np.float32) / np.sqrt(dim)
    sbert = _unit(cap_i.reshape(-1, dim) @ proj + 0.1 * rng.normal(size=(n_img * 5, 384)))
    save_embeddings(
        E / "sbert_txt_flickr8k.npy",
        sbert,
        [f"{i}_c{j}" for i in img_ids for j in range(5)],
        dim=384,
    )
    save_embeddings(E / "audio_esc50.npy", esc, [f"esc_{i:06d}" for i in range(n_esc)])
    return root
