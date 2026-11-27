"""Extraction des embeddings sur GPU (Colab / Kaggle) avec sauvegarde incrémentale et reprise.

Étapes (--stage) :
  text       CLIP-T (ou SBERT avec --encoder sbert) sur les légendes d'un dataset
  image      CLIP-V sur les images de Flickr8k
  audio_raw  AST brut (768 d, non normalisé) sur Clotho ou ESC-50
  audio_proj applique la tête entraînée (audio_proj.pt) : ast_raw_<ds>.npy -> audio_<ds>.npy
Chaque bloc de `--chunk` éléments est écrit dans <fichier>.parts/ ; relancer la commande reprend
là où une session coupée s'est arrêtée (risque n°2 du plan de risques).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from mmsearch.config import load_config, resolve_device, set_seed
from mmsearch.data.loaders import load_captions, load_manifest
from mmsearch.embeddings_io import load_embeddings, save_embeddings


def encode_in_chunks(
    encode_fn: Callable[[Sequence], np.ndarray],
    items: Sequence,
    out: Path,
    chunk: int = 500,
    keep_parts: bool = False,
) -> np.ndarray:
    parts = out.with_suffix(".parts")
    parts.mkdir(parents=True, exist_ok=True)
    n_parts = (len(items) + chunk - 1) // chunk
    for c in range(n_parts):
        f = parts / f"{c:05d}.npy"
        if f.exists():
            continue
        np.save(f, encode_fn(items[c * chunk : (c + 1) * chunk]), allow_pickle=False)
        print(f"bloc {c + 1}/{n_parts} écrit", flush=True)
    arr = np.concatenate([np.load(parts / f"{c:05d}.npy") for c in range(n_parts)])
    if not keep_parts:
        shutil.rmtree(parts)
    return arr


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None)
    ap.add_argument("--stage", required=True, choices=["text", "image", "audio_raw", "audio_proj"])
    ap.add_argument("--dataset", required=True, choices=["flickr8k", "clotho", "esc50"])
    ap.add_argument("--encoder", default="clip", choices=["clip", "sbert"])
    ap.add_argument("--chunk", type=int, default=500)
    ap.add_argument("--fp16", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    set_seed(cfg.seed)
    dev = resolve_device(cfg["device"])
    man = load_manifest(cfg)
    man = man[man["dataset"] == a.dataset]
    E = cfg.emb_dir

    if a.stage == "text":
        caps = load_captions(cfg)
        caps = caps[caps["item_id"].isin(set(man["item_id"]))].sort_values(["item_id", "idx"])
        ids, texts = caps["caption_id"].tolist(), caps["text"].tolist()
        if a.encoder == "clip":
            from mmsearch.encoders.text import ClipTextEncoder

            enc, name, dim = (
                ClipTextEncoder(cfg["models"]["clip"], dev),
                f"clip_txt_{a.dataset}.npy",
                cfg.emb_dim,
            )
        else:
            from mmsearch.encoders.text import SbertTextEncoder

            enc, name, dim = (
                SbertTextEncoder(cfg["models"]["sbert"], dev),
                f"sbert_txt_{a.dataset}.npy",
                384,
            )
        arr = encode_in_chunks(enc.encode, texts, E / name, a.chunk)
        save_embeddings(E / name, arr, ids, dim=dim)
    elif a.stage == "image":
        from mmsearch.encoders.image import ClipImageEncoder

        m = man[man["modality"] == "image"].sort_values("item_id")
        paths = [str(cfg.root / p) for p in m["path"]]
        enc = ClipImageEncoder(cfg["models"]["clip"], dev, fp16=a.fp16)
        name = f"clip_img_{a.dataset}.npy"
        save_embeddings(
            E / name, encode_in_chunks(enc.encode, paths, E / name, a.chunk), m["item_id"].tolist()
        )
    elif a.stage == "audio_raw":
        from mmsearch.encoders.audio import AST_DIM, AstBackbone

        m = man[man["modality"] == "audio"].sort_values("item_id")
        paths = [str(cfg.root / p) for p in m["path"]]
        a_cfg = cfg["audio"]
        enc = AstBackbone(
            cfg["models"]["ast"],
            dev,
            a_cfg["sample_rate"],
            a_cfg["window_s"],
            a_cfg["min_tail_s"],
            max_seconds=a_cfg["max_upload_s"],
        )
        name = f"ast_raw_{a.dataset}.npy"
        save_embeddings(
            E / name,
            encode_in_chunks(
                lambda batch: enc.encode_raw(batch, batch_size=4),
                paths,
                E / name,
                max(1, a.chunk // 10),
            ),
            m["item_id"].tolist(),
            dim=AST_DIM,
            normalized=False,
        )
    else:
        from mmsearch.encoders.audio import AST_DIM, project_raw

        raw, ids = load_embeddings(E / f"ast_raw_{a.dataset}.npy", dim=AST_DIM, normalized=False)
        save_embeddings(
            E / f"audio_{a.dataset}.npy",
            project_raw(raw, cfg.model_dir / "audio_proj.pt", dev),
            ids,
        )


if __name__ == "__main__":
    main()
