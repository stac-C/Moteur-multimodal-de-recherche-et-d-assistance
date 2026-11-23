"""Génère toutes les figures du rapport et des slides à partir des fichiers produits par les scripts.

Aucune figure n'est faite à la main : chaque fonction lit un JSON ou un .npy d'artifacts/ et écrit un PNG
dans artifacts/figures/. Les figures dont l'entrée est absente sont ignorées (message affiché).
Lancer : python -m mmsearch.eval.make_figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from mmsearch.config import Config, load_config  # noqa: E402

COL = {"image": "#2f6fdb", "caption": "#2e9d57", "audio": "#e08a1e"}


def _json(path: Path) -> Optional[dict]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def fig_mono_vs_multi(cfg: Config, out: Path) -> Optional[Path]:
    res = _json(out / "results.json")
    if not res:
        return None
    systems = res["texte->image (Flickr8k)"]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    width, x = 0.8 / len(systems), np.arange(3)
    for i, (name, m) in enumerate(systems.items()):
        ax.bar(x + i * width, [100 * m[k] for k in ("R@1", "R@5", "R@10")], width, label=name)
    ax.set_xticks(x + 0.4 - width / 2)
    ax.set_xticklabels(["R@1", "R@5", "R@10"])
    ax.set_ylabel("Recall (%)")
    ax.set_title("Texte → image : mono-modal contre multimodal")
    ax.legend(fontsize=7)
    return _save(fig, out / "fig_mono_vs_multi.png")


def fig_fusion_weights(cfg: Config, out: Path) -> Optional[Path]:
    fw = _json(out / "fusion_weights.json")
    if not fw:
        return None
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for name, d in fw.items():
        ax.plot(
            [r["w_mod"] for r in d["table"]],
            [r["rsum"] for r in d["table"]],
            marker="o",
            label=name,
        )
        ax.axvline(d["best_w_mod"], ls=":", lw=0.8, color="grey")
    ax.set_xlabel("poids de la modalité propre w (légendes : 1 − w)")
    ax.set_ylabel("rsum = R@1 + R@5 + R@10 (validation)")
    ax.set_title("Réglage des poids de la fusion tardive")
    ax.legend()
    return _save(fig, out / "fig_fusion_weights.png")


def fig_dim_ablation(cfg: Config, out: Path) -> Optional[Path]:
    res = _json(out / "results.json")
    if not res:
        return None
    ab = res["ablation dimension d (fusion tardive)"]
    dims = sorted(ab, key=int)
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for k in ("R@1", "R@5", "R@10"):
        ax.plot([int(d) for d in dims], [100 * ab[d][k] for d in dims], marker="o", label=k)
    ax.set_xscale("log", base=2)
    ax.set_xticks([int(d) for d in dims])
    ax.set_xticklabels(dims)
    ax.set_xlabel("dimension d")
    ax.set_ylabel("Recall (%)")
    ax.set_title("Ablation de la dimension (texte → image)")
    ax.legend()
    return _save(fig, out / "fig_dim_ablation.png")


def fig_audio_training(cfg: Config, out: Path) -> Optional[Path]:
    log = _json(cfg.model_dir / "audio_proj_log.json")
    if not log:
        return None
    h = log["history"]
    ep = [r["epoch"] for r in h]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(ep, [r["loss"] for r in h], color="#c0392b", label="perte InfoNCE (train)")
    ax.set_xlabel("époque")
    ax.set_ylabel("perte")
    ax2 = ax.twinx()
    ax2.plot(ep, [100 * r["a2t_R@10"] for r in h], color="#2f6fdb", label="audio→texte R@10 (val)")
    ax2.set_ylabel("Recall@10 (%)")
    ax.set_title("Entraînement de la tête audio")
    fig.legend(loc="center right", fontsize=7)
    return _save(fig, out / "fig_audio_training.png")


def fig_zero_shot(cfg: Config, out: Path) -> list[Path]:
    res, cm = _json(out / "zero_shot_esc50.json"), out / "zero_shot_esc50_confusion.npy"
    paths = []
    if res:
        items = sorted(res["per_class"].items(), key=lambda kv: kv[1])
        fig, ax = plt.subplots(figsize=(6, max(4, 0.17 * len(items))))
        ax.barh(
            [k.replace("_", " ") for k, _ in items], [100 * v for _, v in items], color="#7b3fa0"
        )
        ax.axvline(100 * res["chance"], color="grey", ls=":")
        ax.set_xlabel("exactitude (%)")
        ax.set_title(
            f"ESC-50 zero-shot : top-1 {100 * res['top1']:.1f} %, top-5 {100 * res['top5']:.1f} %"
        )
        ax.tick_params(axis="y", labelsize=6)
        paths.append(_save(fig, out / "fig_zero_shot_per_class.png"))
    if res and cm.exists():
        m = np.load(cm).astype(float)
        m = m / np.maximum(m.sum(axis=1, keepdims=True), 1)
        fig, ax = plt.subplots(figsize=(7, 6.2))
        im = ax.imshow(m, cmap="Blues", vmin=0, vmax=1)
        names = [k.replace("_", " ") for k in sorted(res["per_class"])]
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=90, fontsize=5)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=5)
        ax.set_xlabel("classe prédite")
        ax.set_ylabel("classe réelle")
        fig.colorbar(im, ax=ax, fraction=0.04)
        paths.append(_save(fig, out / "fig_zero_shot_confusion.png"))
    return paths


def fig_projections(cfg: Config, out: Path) -> list[Path]:
    paths = []
    for method in ("pca", "tsne", "umap"):
        proj = _json(cfg.artifacts / "projections" / f"{method}_joint.json")
        if not proj:
            continue
        fig, ax = plt.subplots(figsize=(6, 5))
        for mod, c in COL.items():
            pts = [p for p in proj["points"] if p["modality"] == mod]
            ax.scatter([p["x"] for p in pts], [p["y"] for p in pts], s=5, alpha=0.6, c=c, label=mod)
        ax.set_title(f"Espace commun projeté en 2D ({method.upper()})")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(markerscale=3)
        paths.append(_save(fig, out / f"fig_proj_{method}.png"))
    return paths


def fig_bias(cfg: Config, out: Path) -> Optional[Path]:
    rep = _json(out / "bias_report.json")
    if not rep:
        return None
    g = {k: v for k, v in rep["retrieval"]["groups"].items() if v["n_queries"] >= 30}
    names = list(g)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for i, system in enumerate(("fusion", "image seule")):
        ax.bar(
            x + i * 0.4,
            [100 * g[n][f"{system} R@10"] for n in names],
            0.4,
            yerr=[100 * g[n][f"{system} R@10 ci95"] for n in names],
            capsize=2,
            label=system,
        )
    ax.set_xticks(x + 0.2)
    ax.set_xticklabels([f"{n}\n(n={g[n]['n_queries']})" for n in names], fontsize=7)
    ax.set_ylabel("Recall@10 (%)")
    ax.set_title("Texte → image par groupe thématique de requêtes")
    ax.legend()
    return _save(fig, out / "fig_bias_groups.png")


def main(cfg: Config) -> list[Path]:
    out = cfg.artifacts / "figures"
    out.mkdir(parents=True, exist_ok=True)
    made: list[Path] = []
    for fn in (
        fig_mono_vs_multi,
        fig_fusion_weights,
        fig_dim_ablation,
        fig_audio_training,
        fig_zero_shot,
        fig_projections,
        fig_bias,
    ):
        r = fn(cfg, out)
        for p in r if isinstance(r, list) else [r]:
            if p is None:
                print(f"ignorée : {fn.__name__} (entrée absente)")
            else:
                made.append(p)
                print("écrit", p.name)
    return made


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None)
    main(load_config(ap.parse_args().config))
