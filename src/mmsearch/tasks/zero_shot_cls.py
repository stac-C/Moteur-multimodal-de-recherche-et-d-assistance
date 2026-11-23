"""Tâche secondaire : classification audio zero-shot sur ESC-50 (section 1.6).

Prédiction : y_hat = argmax_c  z_audio . z_{t_c}, avec t_c = « the sound of {classe} » (CLIP texte).
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from mmsearch.config import load_config, resolve_device
from mmsearch.data.loaders import load_manifest
from mmsearch.embeddings_io import load_embeddings

TEMPLATE = "the sound of {}"


def class_prompts(classes: list[str], template: str = TEMPLATE) -> list[str]:
    """Les noms ESC-50 utilisent des underscores (ex. pouring_water) : remplacés par des espaces."""
    return [template.format(c.replace("_", " ")) for c in classes]


def zero_shot_scores(audio_emb: np.ndarray, class_emb: np.ndarray) -> np.ndarray:
    """Cosinus (n, C) entre clips et classes (vecteurs normalisés)."""
    return audio_emb @ class_emb.T


def topk_accuracy(scores: np.ndarray, y_true: np.ndarray, k: int = 1) -> float:
    top = np.argsort(-scores, axis=1)[:, :k]
    return float((top == y_true[:, None]).any(axis=1).mean())


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def classify_one(
    audio_emb: np.ndarray,
    class_emb: np.ndarray,
    classes: list[str],
    topk: int = 5,
    scale: float = 100.0,
) -> list[tuple[str, float]]:
    """Distribution softmax(scale * cos) pour un clip : utilisé par /api/classify."""
    logits = scale * (class_emb @ audio_emb.reshape(-1))
    p = np.exp(logits - logits.max())
    p /= p.sum()
    order = np.argsort(-p)[:topk]
    return [(classes[i], float(p[i])) for i in order]


def main() -> None:
    from mmsearch.encoders.text import ClipTextEncoder

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    man = load_manifest(cfg)
    man = man[man["dataset"] == "esc50"]
    emb, ids = load_embeddings(cfg.emb_dir / "audio_esc50.npy")
    labels = man.set_index("item_id").loc[ids, "label"].tolist()
    classes = sorted(set(labels))
    y = np.array([classes.index(lab) for lab in labels])
    enc = ClipTextEncoder(cfg["models"]["clip"], resolve_device(cfg["device"]))
    class_emb = enc.encode(class_prompts(classes))
    s = zero_shot_scores(emb, class_emb)
    pred = s.argmax(1)
    cm = confusion_matrix(y, pred, len(classes))
    res = {
        "n_clips": len(y),
        "n_classes": len(classes),
        "top1": topk_accuracy(s, y, 1),
        "top5": topk_accuracy(s, y, 5),
        "chance": 1.0 / len(classes),
        "per_class": {c: float(cm[i, i] / max(cm[i].sum(), 1)) for i, c in enumerate(classes)},
    }
    out = cfg.artifacts / "figures"
    out.mkdir(parents=True, exist_ok=True)
    (out / "zero_shot_esc50.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    np.save(out / "zero_shot_esc50_confusion.npy", cm)
    print(f"top-1 = {res['top1']:.3f}  top-5 = {res['top5']:.3f}  (hasard = {res['chance']:.3f})")


if __name__ == "__main__":
    main()
