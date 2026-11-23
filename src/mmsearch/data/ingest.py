"""Ingestion de Flickr8k, Clotho v2 et ESC-50 : produit manifest.csv et captions.csv.

Arborescence attendue sous data/raw/ (vérifier les noms de fichiers après téléchargement) :
  flickr8k/Images/*.jpg, flickr8k/Flickr8k.token.txt,
  flickr8k/Flickr_8k.{trainImages,devImages,testImages}.txt
  clotho/{development,validation,evaluation}/*.wav,
  clotho/clotho_captions_{development,validation,evaluation}.csv
  esc50/audio/*.wav, esc50/meta/esc50.csv

Schéma du manifest : item_id, modality, dataset, split, path, label, duration_s
Schéma des légendes : caption_id, item_id, idx, text
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mmsearch.config import Config, load_config
from mmsearch.data.preprocess import clean_text

MANIFEST_COLS = ["item_id", "modality", "dataset", "split", "path", "label", "duration_s"]


def _rel(cfg: Config, p: Path) -> str:
    return p.resolve().relative_to(cfg.root).as_posix()


def ingest_flickr8k(cfg: Config) -> tuple[list[dict], list[dict]]:
    base = cfg.raw / "flickr8k"
    split_files = {
        "train": "Flickr_8k.trainImages.txt",
        "val": "Flickr_8k.devImages.txt",
        "test": "Flickr_8k.testImages.txt",
    }
    split_of: dict[str, str] = {}
    for split, fname in split_files.items():
        for line in (base / fname).read_text(encoding="utf-8").splitlines():
            if line.strip():
                split_of[line.strip()] = split
    caps: dict[str, list[str]] = {}
    for line in (base / "Flickr8k.token.txt").read_text(encoding="utf-8").splitlines():
        if "\t" not in line:
            continue
        key, text = line.split("\t", 1)
        caps.setdefault(key.split("#")[0], []).append(clean_text(text))
    items, captions = [], []
    for n, fname in enumerate(sorted(split_of)):
        item_id = f"img_{n:06d}"
        items.append(
            dict(
                item_id=item_id,
                modality="image",
                dataset="flickr8k",
                split=split_of[fname],
                path=_rel(cfg, base / "Images" / fname),
                label="",
                duration_s="",
            )
        )
        for j, text in enumerate(caps[fname][:5]):
            captions.append(dict(caption_id=f"{item_id}_c{j}", item_id=item_id, idx=j, text=text))
    return items, captions


def ingest_clotho(cfg: Config) -> tuple[list[dict], list[dict]]:
    import soundfile as sf

    base = cfg.raw / "clotho"
    items, captions, n = [], [], 0
    for split, folder in {
        "train": "development",
        "val": "validation",
        "test": "evaluation",
    }.items():
        df = pd.read_csv(base / f"clotho_captions_{folder}.csv", dtype=str, keep_default_na=False)
        for _, row in df.iterrows():
            wav = base / folder / row["file_name"]
            item_id = f"aud_{n:06d}"
            n += 1
            items.append(
                dict(
                    item_id=item_id,
                    modality="audio",
                    dataset="clotho",
                    split=split,
                    path=_rel(cfg, wav),
                    label="",
                    duration_s=f"{sf.info(str(wav)).duration:.2f}",
                )
            )
            for j in range(5):
                captions.append(
                    dict(
                        caption_id=f"{item_id}_c{j}",
                        item_id=item_id,
                        idx=j,
                        text=clean_text(row[f"caption_{j + 1}"]),
                    )
                )
    return items, captions


def ingest_esc50(cfg: Config) -> list[dict]:
    base = cfg.raw / "esc50"
    df = pd.read_csv(base / "meta" / "esc50.csv", dtype=str, keep_default_na=False)
    items = []
    for n, (_, row) in enumerate(df.sort_values("filename").iterrows()):
        items.append(
            dict(
                item_id=f"esc_{n:06d}",
                modality="audio",
                dataset="esc50",
                split="test",
                path=_rel(cfg, base / "audio" / row["filename"]),
                label=row["category"],
                duration_s="5.00",
            )
        )
    return items


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None)
    ap.add_argument("--datasets", nargs="+", default=["flickr8k", "clotho", "esc50"])
    args = ap.parse_args()
    cfg = load_config(args.config)
    items: list[dict] = []
    captions: list[dict] = []
    if "flickr8k" in args.datasets:
        i, c = ingest_flickr8k(cfg)
        items += i
        captions += c
    if "clotho" in args.datasets:
        i, c = ingest_clotho(cfg)
        items += i
        captions += c
    if "esc50" in args.datasets:
        items += ingest_esc50(cfg)
    cfg.processed.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(items, columns=MANIFEST_COLS).to_csv(cfg.processed / "manifest.csv", index=False)
    pd.DataFrame(captions, columns=["caption_id", "item_id", "idx", "text"]).to_csv(
        cfg.processed / "captions.csv", index=False
    )
    print(f"{len(items)} items, {len(captions)} légendes écrits dans {cfg.processed}")


if __name__ == "__main__":
    main()
