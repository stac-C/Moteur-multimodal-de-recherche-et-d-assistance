"""Évaluation complète : toutes les expériences du rapport, écrites dans artifacts/figures/.

Sorties : results.json, results.csv, results_tables.tex (booktabs), case_studies.json.
Protocole (sections 1.5 et 1.6) : leave-one-out pour tout système qui voit les légendes de l'item.
Lancer : python -m mmsearch.eval.run_eval --split test
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from typing import Callable, Optional

import numpy as np

from mmsearch.config import Config, load_config
from mmsearch.data.loaders import captions_tensor, load_captions, load_manifest
from mmsearch.embeddings_io import load_embeddings
from mmsearch.encoders.base import l2_normalize
from mmsearch.eval.metrics import (
    first_relevant_rank,
    precision_from_scores,
    ranks_from_scores,
    summary,
)
from mmsearch.fusion.late import (
    caption_context,
    evaluate_text_to_item,
    text_to_item_scores_loo,
)

KS = (1, 5, 10)
KEYWORDS = {
    "dog": ["dog", "dogs", "puppy"],
    "cat": ["cat", "cats", "kitten"],
    "cow": ["cow", "cows"],
    "pig": ["pig", "pigs"],
    "hen": ["hen", "chicken", "chickens"],
    "sheep": ["sheep", "lamb"],
    "frog": ["frog", "frogs"],
    "crow": ["crow", "crows"],
    "rooster": ["rooster", "roosters"],
    "chirping_birds": ["bird", "birds"],
    "train": ["train", "trains"],
    "airplane": ["airplane", "airplanes", "plane", "jet"],
    "helicopter": ["helicopter"],
    "sea_waves": ["wave", "waves", "ocean", "sea"],
    "rain": ["rain"],
    "fireworks": ["firework", "fireworks"],
    "crying_baby": ["baby", "infant"],
    "laughing": ["laugh", "laughing"],
    "clapping": ["clap", "clapping"],
    "church_bells": ["church"],
    "keyboard_typing": ["keyboard", "typing"],
}


# ----------------------------------------------------------------------------- chargement
def _rows(ids: list[str], split_of: dict[str, str], split: str) -> np.ndarray:
    return np.array([i for i, it in enumerate(ids) if split_of[it] == split])


def load_flickr(cfg: Config, split: str) -> dict:
    man, caps = load_manifest(cfg), load_captions(cfg)
    split_of = man.set_index("item_id")["split"].to_dict()
    img, ids = load_embeddings(cfg.emb_dir / "clip_img_flickr8k.npy")
    cap_emb, cap_ids = load_embeddings(cfg.emb_dir / "clip_txt_flickr8k.npy")
    sel = [ids[i] for i in _rows(ids, split_of, split)]
    pos = {it: i for i, it in enumerate(ids)}
    out = {
        "ids": sel,
        "img": img[[pos[i] for i in sel]],
        "cap": captions_tensor(sel, caps, cap_emb, cap_ids),
    }
    sb = cfg.emb_dir / "sbert_txt_flickr8k.npy"
    if sb.exists():
        sb_emb, sb_ids = load_embeddings(sb, dim=384)
        out["sbert"] = captions_tensor(sel, caps, sb_emb, sb_ids)
    texts = caps.sort_values(["item_id", "idx"]).groupby("item_id")["text"].apply(list).to_dict()
    out["texts"] = [texts[i] for i in sel]
    return out


def early_fusion_fn(cfg: Config) -> Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]]:
    path = cfg.model_dir / "early_fusion.pt"
    if not path.exists():
        return None
    try:
        import torch

        from mmsearch.fusion.early import EarlyFusion
    except ImportError:
        return None
    ck = torch.load(path, map_location="cpu", weights_only=True)
    model = EarlyFusion(cfg.emb_dim, ck["w_mod_init"], 1.0 - ck["w_mod_init"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return lambda a, b: model.fuse_numpy(a, b, "cpu")


def late(w: float) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    return lambda a, b: w * a + (1.0 - w) * b


# ----------------------------------------------------------------------------- expériences
def text_to_image(cfg: Config, fl: dict, w: float) -> dict[str, dict]:
    """Texte -> image : mono-modal vs multimodal (protocole 1.6)."""
    n, K, _ = fl["cap"].shape
    owner = np.repeat(np.arange(n), K)
    res: dict[str, dict] = {}
    if "sbert" in fl:
        full, loo = caption_context(fl["sbert"])
        s = text_to_item_scores_loo(
            fl["sbert"].reshape(n * K, -1), full, loo.reshape(n * K, -1), owner
        )
        res["mono-texte (SBERT, leave-one-out)"] = summary(ranks_from_scores(s, owner))
    res["mono-texte (CLIP, leave-one-out)"] = evaluate_text_to_item(
        lambda a, b: b, fl["img"], fl["cap"]
    )
    res["mono-image (CLIP)"] = evaluate_text_to_item(lambda a, b: a, fl["img"], fl["cap"])
    res[f"fusion tardive (w_img = {w:.1f})"] = evaluate_text_to_item(late(w), fl["img"], fl["cap"])
    ef = early_fusion_fn(cfg)
    if ef is not None:
        res["fusion précoce (MLP)"] = evaluate_text_to_item(ef, fl["img"], fl["cap"])
    return res


def pair_metrics(a: np.ndarray, cap: np.ndarray) -> dict[str, dict]:
    """Recherche bidirectionnelle entre n items (a : (n, d)) et leurs 5 légendes (cap : (n, K, d))."""
    n, K, _ = cap.shape
    S = a @ cap.reshape(n * K, -1).T  # (n, nK)
    owner = np.repeat(np.arange(n), K)
    rel = owner[None, :] == np.arange(n)[:, None]
    i2t = summary(first_relevant_rank(S, rel))
    for k in KS:
        i2t[f"P@{k}"] = precision_from_scores(S, rel, k)
    t2i = summary(ranks_from_scores(np.ascontiguousarray(S.T), owner))
    return {"item->texte": i2t, "texte->item": t2i}


def audio_text(cfg: Config, split: str) -> dict[str, dict]:
    """Audio <-> texte sur Clotho (paires natives) et fusion audio + légendes (texte -> audio)."""
    man, caps = load_manifest(cfg), load_captions(cfg)
    split_of = man.set_index("item_id")["split"].to_dict()
    aud, ids = load_embeddings(cfg.emb_dir / "audio_clotho.npy")
    cap_emb, cap_ids = load_embeddings(cfg.emb_dir / "clip_txt_clotho.npy")
    rows = _rows(ids, split_of, split)
    sel = [ids[i] for i in rows]
    cap = captions_tensor(sel, caps, cap_emb, cap_ids)
    wa = cfg["fusion"]["weights_audio"]["audio"]
    res = {"AST + MLP (E5)": pair_metrics(aud[rows], cap)}
    res["texte->audio, audio seul"] = {
        "texte->item": evaluate_text_to_item(lambda a, b: a, aud[rows], cap)
    }
    res[f"texte->audio, fusion tardive (w_aud = {wa:.1f})"] = {
        "texte->item": evaluate_text_to_item(late(wa), aud[rows], cap)
    }
    ca, cb = cfg.emb_dir / "clap_audio_clotho.npy", cfg.emb_dir / "clap_txt_clotho.npy"
    if ca.exists() and cb.exists():  # baseline LAION-CLAP (tâche C5)
        c_aud, c_ids = load_embeddings(ca)
        c_cap, c_cids = load_embeddings(cb)
        pos = {i: n for n, i in enumerate(c_ids)}
        res["LAION-CLAP (zero-shot)"] = pair_metrics(
            c_aud[[pos[i] for i in sel]], captions_tensor(sel, caps, c_cap, c_cids)
        )
    return res


def pca_fit(X: np.ndarray, d: int) -> tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    _, _, vt = np.linalg.svd(X - mu, full_matrices=False)
    return mu, vt[:d]


def dim_ablation(train: dict, test: dict, w: float, dims=(128, 256, 512)) -> dict[str, dict]:
    """Texte -> image en fusion tardive après réduction PCA (ajustée sur train) à d dimensions.

    Si l'échantillon d'ajustement a moins de d lignes (jeux factices), d est ramené au rang disponible.
    """
    out = {}
    D = train["img"].shape[1]
    X = np.concatenate([train["img"], train["cap"].reshape(-1, D)])
    for d in dims:
        if d >= D:
            img, cap = test["img"], test["cap"]
        else:
            mu, comps = pca_fit(X, d)
            f = lambda Z, mu=mu, comps=comps: l2_normalize((Z - mu) @ comps.T)  # noqa: E731
            img = f(test["img"])
            cap = f(test["cap"].reshape(-1, D)).reshape(*test["cap"].shape[:2], comps.shape[0])
        out[str(d)] = evaluate_text_to_item(late(w), img, cap)
    return out


def proxy_precision(
    S: np.ndarray,
    labels: list[str],
    texts: list[str],
    keywords: dict[str, list[str]],
    k: int = 10,
    min_relevant: int = 5,
) -> dict:
    """Audio -> image sans vérité terrain : une image est pertinente si une légende contient un mot-clé.

    S (n_clips, n_images) ; labels : classe de chaque clip ; texts : légendes concaténées par image.
    Retourne la précision@k par classe, le taux de base (part d'images pertinentes) et le gain (lift).
    """
    top = np.argsort(-S, axis=1)[:, :k]
    per = {}
    for cls, kws in keywords.items():
        idx = [i for i, lab in enumerate(labels) if lab == cls]
        pat = re.compile(r"\b(" + "|".join(map(re.escape, kws)) + r")\b")
        rel = np.array([bool(pat.search(t)) for t in texts])
        if not idx or rel.sum() < min_relevant:
            continue
        p = float(np.mean([rel[top[i]].mean() for i in idx]))
        per[cls] = {
            f"P@{k}": p,
            "base_rate": float(rel.mean()),
            "lift": p / float(rel.mean()),
            "n_relevant_images": int(rel.sum()),
            "n_clips": len(idx),
        }
    macro = None
    if per:
        macro = {
            key: float(np.mean([v[key] for v in per.values()]))
            for key in (f"P@{k}", "base_rate", "lift")
        }
    return {"classes": per, "macro": macro}


def audio_to_image(cfg: Config, k: int = 10) -> dict:
    man, caps = load_manifest(cfg), load_captions(cfg)
    esc, esc_ids = load_embeddings(cfg.emb_dir / "audio_esc50.npy")
    img, img_ids = load_embeddings(cfg.emb_dir / "clip_img_flickr8k.npy")
    labels = man.set_index("item_id").loc[esc_ids, "label"].tolist()
    by_img = caps.groupby("item_id")["text"].apply(lambda s: " ".join(s).lower()).to_dict()
    return proxy_precision(esc @ img.T, labels, [by_img.get(i, "") for i in img_ids], KEYWORDS, k)


def case_studies(fl: dict, w: float, n_each: int = 5, seed: int = 42) -> list[dict]:
    """Requêtes texte -> image : succès (rang 1) et échecs (rang > 10) pour l'étude de cas."""
    n, K, _ = fl["cap"].shape
    ctx_full, ctx_loo = caption_context(fl["cap"])
    owner = np.repeat(np.arange(n), K)
    z_full = w * fl["img"] + (1 - w) * ctx_full
    z_loo = w * fl["img"][owner] + (1 - w) * ctx_loo.reshape(n * K, -1)
    S = text_to_item_scores_loo(fl["cap"].reshape(n * K, -1), z_full, z_loo, owner)
    ranks = ranks_from_scores(S, owner)
    rng = np.random.default_rng(seed)
    out = []
    for name, mask in (("succès", ranks == 1), ("échec", ranks > 10)):
        cand = np.flatnonzero(mask)
        for q in rng.permutation(cand)[:n_each]:
            out.append(
                {
                    "outcome": name,
                    "query": fl["texts"][q // K][q % K],
                    "gold": fl["ids"][owner[q]],
                    "rank": int(ranks[q]),
                    "top5": [fl["ids"][j] for j in np.argsort(-S[q])[:5]],
                }
            )
    return out


# ----------------------------------------------------------------------------- sorties
def _fmt(col: str, v: float) -> str:
    if col.startswith(("R@", "P@")):
        return f"{100 * v:.1f}"
    return f"{v:.3f}" if col == "MRR" else f"{v:.0f}"


def _tex(s: str) -> str:
    return (
        s.replace("\\", "")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("->", "$\\to$")
    )


def latex_table(rows: dict[str, dict], cols: list[str], caption: str, label: str) -> str:
    head = " & ".join(["Système"] + [c.replace("MedR", "Rang méd.") for c in cols])
    body = [
        " & ".join([_tex(name)] + [_fmt(c, m[c]) for c in cols]) + r" \\"
        for name, m in rows.items()
    ]
    return "\n".join(
        [
            r"\begin{table}[ht]\centering",
            r"\begin{tabular}{l" + "r" * len(cols) + "}",
            r"\toprule",
            head + r" \\",
            r"\midrule",
            *body,
            r"\bottomrule",
            r"\end{tabular}",
            rf"\caption{{{_tex(caption)}}}\label{{{label}}}",
            r"\end{table}",
            "",
        ]
    )


def flatten(results: dict) -> list[tuple]:
    rows = []
    for exp, systems in results.items():
        if exp in ("audio->image (proxy)", "meta"):
            continue
        for system, m in systems.items():
            sub = m if all(isinstance(v, (int, float)) for v in m.values()) else None
            for name, metrics in [("", m)] if sub else m.items():
                for key, val in metrics.items():
                    rows.append((exp, f"{system} {name}".strip(), key, val))
    return rows


def run(cfg: Config, split: str = "test", w: Optional[float] = None) -> dict:
    w = cfg["fusion"]["weights_image"]["image"] if w is None else w
    test, train = load_flickr(cfg, split), load_flickr(cfg, "train")
    results: dict = {
        "meta": {
            "split": split,
            "n_images": len(test["ids"]),
            "n_queries_text": int(test["cap"].shape[0] * test["cap"].shape[1]),
            "w_img": w,
        }
    }
    results["texte->image (Flickr8k)"] = text_to_image(cfg, test, w)
    results["image<->texte (CLIP, Flickr8k)"] = {
        "CLIP image <-> légendes": pair_metrics(test["img"], test["cap"])
    }
    results["ablation dimension d (fusion tardive)"] = dim_ablation(train, test, w)
    if (cfg.emb_dir / "audio_clotho.npy").exists():
        results["audio<->texte (Clotho)"] = audio_text(cfg, split)
    if (cfg.emb_dir / "audio_esc50.npy").exists():
        results["audio->image (proxy)"] = audio_to_image(cfg)
    out = cfg.artifacts / "figures"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with open(out / "results.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["experience", "systeme", "metrique", "valeur"])
        wr.writerows(flatten(results))
    cols = ["R@1", "R@5", "R@10", "MRR", "MedR"]
    tex = [
        latex_table(
            results["texte->image (Flickr8k)"],
            cols,
            "Texte vers image, Flickr8k : mono-modal contre multimodal.",
            "tab:text2image",
        ),
        latex_table(
            {f"d = {d}": m for d, m in results["ablation dimension d (fusion tardive)"].items()},
            cols,
            "Ablation de la dimension d de l'espace commun.",
            "tab:dim",
        ),
    ]
    (out / "results_tables.tex").write_text("\n".join(tex), encoding="utf-8")
    (out / "case_studies.json").write_text(
        json.dumps(case_studies(test, w), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--w", type=float, default=None, help="poids w_img (défaut : config.yaml)")
    a = ap.parse_args()
    res = run(load_config(a.config), a.split, a.w)
    for system, m in res["texte->image (Flickr8k)"].items():
        print(
            f"{system:42s} R@1 {100 * m['R@1']:5.1f}  R@5 {100 * m['R@5']:5.1f}  R@10 {100 * m['R@10']:5.1f}"
        )
