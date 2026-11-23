"""Règle les poids de la fusion tardive sur la validation (livrable B3). Écrit fusion_weights.json."""

from __future__ import annotations

import argparse
import json

import numpy as np

from mmsearch.config import load_config
from mmsearch.data.loaders import captions_tensor, load_captions, load_manifest
from mmsearch.embeddings_io import load_embeddings
from mmsearch.fusion.late import grid_search_weights


def split_rows(ids: list[str], man_split: dict[str, str], split: str) -> np.ndarray:
    return np.array([i for i, it in enumerate(ids) if man_split[it] == split])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default="val")
    args = ap.parse_args()
    cfg = load_config(args.config)
    man, caps = load_manifest(cfg), load_captions(cfg)
    sp = man.set_index("item_id")["split"].to_dict()
    out = {}
    for name, mod_file, cap_file in (
        ("image", "clip_img_flickr8k.npy", "clip_txt_flickr8k.npy"),
        ("audio", "audio_clotho.npy", "clip_txt_clotho.npy"),
    ):
        mod, ids = load_embeddings(cfg.emb_dir / mod_file)
        cap_emb, cap_ids = load_embeddings(cfg.emb_dir / cap_file)
        rows = split_rows(ids, sp, args.split)
        cap = captions_tensor([ids[i] for i in rows], caps, cap_emb, cap_ids)
        w, table = grid_search_weights(mod[rows], cap, step=0.1)
        out[name] = {"best_w_mod": w, "table": table}
        print(
            f"{name}: w_mod = {w:.1f}, w_cap = {1 - w:.1f} -> à reporter dans config.yaml (fusion.weights_{name})"
        )
    path = cfg.artifacts / "figures" / "fusion_weights.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
