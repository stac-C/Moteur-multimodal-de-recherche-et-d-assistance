"""Construit les 4 index FAISS, les vecteurs fusionnés et la base SQLite (étapes 6 et 7, section 2.2).

Entrées  : artifacts/embeddings/{clip_img_flickr8k, clip_txt_flickr8k, audio_clotho, clip_txt_clotho}.npy
Sorties  : artifacts/embeddings/fused_items.npy ; artifacts/index/idx_{img,aud,txt,fused}.faiss ; collection.db
"""

from __future__ import annotations

import argparse
import sqlite3

import numpy as np

from mmsearch.config import Config, load_config, set_seed
from mmsearch.data.loaders import captions_tensor, load_captions, load_manifest
from mmsearch.embeddings_io import load_embeddings, save_embeddings
from mmsearch.fusion.late import caption_context, late_fusion
from mmsearch.index.store import VectorStore


def build_sqlite(cfg: Config) -> None:
    manifest, captions = load_manifest(cfg), load_captions(cfg)
    db = cfg.index_dir / "collection.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    if db.exists():
        db.unlink()
    con = sqlite3.connect(db)
    con.execute(
        "CREATE TABLE items (item_id TEXT PRIMARY KEY, modality TEXT, dataset TEXT, split TEXT, "
        "path TEXT, label TEXT, duration_s TEXT)"
    )
    con.execute(
        "CREATE TABLE captions (caption_id TEXT PRIMARY KEY, item_id TEXT, idx INTEGER, text TEXT)"
    )
    con.executemany(
        "INSERT INTO items VALUES (?,?,?,?,?,?,?)",
        manifest[
            ["item_id", "modality", "dataset", "split", "path", "label", "duration_s"]
        ].values.tolist(),
    )
    con.executemany(
        "INSERT INTO captions VALUES (?,?,?,?)",
        captions[["caption_id", "item_id", "idx", "text"]].values.tolist(),
    )
    con.execute("CREATE INDEX idx_captions_item ON captions(item_id)")
    con.commit()
    con.close()


def build_all(cfg: Config) -> dict[str, int]:
    set_seed(cfg.seed)
    E, captions = cfg.emb_dir, load_captions(cfg)
    img, img_ids = load_embeddings(E / "clip_img_flickr8k.npy")
    capf, capf_ids = load_embeddings(E / "clip_txt_flickr8k.npy")
    aud, aud_ids = load_embeddings(E / "audio_clotho.npy")
    capc, capc_ids = load_embeddings(E / "clip_txt_clotho.npy")

    # Fusion tardive : zbar = w_mod * z_mod + w_cap * c  (c = moyenne normalisée des légendes)
    wi, wa = cfg["fusion"]["weights_image"], cfg["fusion"]["weights_audio"]
    ctx_img, _ = caption_context(captions_tensor(img_ids, captions, capf, capf_ids))
    ctx_aud, _ = caption_context(captions_tensor(aud_ids, captions, capc, capc_ids))
    fused_img = late_fusion({"image": img, "caption": ctx_img}, wi)
    fused_aud = late_fusion({"audio": aud, "caption": ctx_aud}, wa)
    fused = np.concatenate([fused_img, fused_aud])
    fused_ids = img_ids + aud_ids
    save_embeddings(E / "fused_items.npy", fused, fused_ids, max_norm=True)

    d = cfg.index_dir
    stores = {
        "idx_img": (img, img_ids),
        "idx_aud": (aud, aud_ids),
        "idx_txt": (np.concatenate([capf, capc]), capf_ids + capc_ids),
        "idx_fused": (fused, fused_ids),
    }
    sizes = {}
    for name, (vec, ids) in stores.items():
        store = VectorStore(cfg.emb_dim)
        store.build(vec, ids)
        store.save(str(d / f"{name}.faiss"))
        sizes[name] = len(store)
    build_sqlite(cfg)
    return sizes


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None)
    print(build_all(load_config(ap.parse_args().config)))
