"""Entraîne la tête audio (AST 768 -> 512) alignée sur l'espace texte CLIP gelé, perte InfoNCE (E5).

Entrées : ast_raw_clotho.npy (768 d, brut), clip_txt_clotho.npy (légendes, ancre gelée).
Sorties : models/audio_proj.pt, models/audio_proj_log.json, embeddings/audio_clotho.npy.
Option --overfit N : entraîne et évalue sur N clips seulement ; le Recall@1 doit tendre vers 1
(test de bon fonctionnement de la chaîne, à faire avant tout balayage d'hyperparamètres).
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from mmsearch.config import load_config, resolve_device, set_seed
from mmsearch.data.loaders import captions_tensor, load_captions, load_manifest
from mmsearch.embeddings_io import load_embeddings, save_embeddings
from mmsearch.encoders.audio import AST_DIM, AudioProjection
from mmsearch.eval.metrics import first_relevant_rank, ranks_from_scores, recall_from_ranks
from mmsearch.training.losses import LOGIT_SCALE_MAX, info_nce

K = 5  # légendes par clip (Clotho)


def load_data(cfg):
    man, caps = load_manifest(cfg), load_captions(cfg)
    raw, ids = load_embeddings(cfg.emb_dir / "ast_raw_clotho.npy", dim=AST_DIM, normalized=False)
    cap_emb, cap_ids = load_embeddings(cfg.emb_dir / "clip_txt_clotho.npy")
    cap = captions_tensor(ids, caps, cap_emb, cap_ids, K)
    split_of = man.set_index("item_id")["split"].to_dict()
    splits = {
        s: np.array([i for i, it in enumerate(ids) if split_of[it] == s])
        for s in ("train", "val", "test")
    }
    return raw, ids, cap, splits


@torch.no_grad()
def project(model: AudioProjection, x: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    return model(torch.from_numpy(x).to(device)).cpu().numpy()


def evaluate(za: np.ndarray, cap: np.ndarray) -> dict[str, float]:
    """Recall@K audio->texte (premier pertinent parmi 5 légendes) et texte->audio ; rsum = somme des 6."""
    n = len(za)
    scores = za @ cap.reshape(n * K, -1).T  # (n, 5n)
    owner = np.repeat(np.arange(n), K)
    a2t = first_relevant_rank(scores, owner[None, :] == np.arange(n)[:, None])
    t2a = ranks_from_scores(np.ascontiguousarray(scores.T), owner)
    m = {
        f"{name}_R@{k}": recall_from_ranks(r, k)
        for name, r in (("a2t", a2t), ("t2a", t2a))
        for k in (1, 5, 10)
    }
    m["rsum"] = float(sum(m.values()))
    return m


def train(cfg, overfit: int = 0) -> dict:
    t = cfg["train_audio"]
    set_seed(cfg.seed)
    device = resolve_device(cfg["device"])
    raw, ids, cap, sp = load_data(cfg)
    tr, va, te = sp["train"], sp["val"], sp["test"]
    if overfit:
        tr = tr[:overfit]
        va = tr
    model = AudioProjection(hidden=t["hidden"], dropout=t["dropout"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    bs = t["batch_size"]
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=t["epochs"] * int(np.ceil(len(tr) / bs))
    )
    rng = np.random.default_rng(cfg.seed)
    X, T = torch.from_numpy(raw).to(device), torch.from_numpy(cap).to(device)
    best, bad, best_state, history = -1.0, 0, None, []
    for epoch in range(t["epochs"]):
        model.train()
        losses = []
        perm = rng.permutation(tr)
        for b in range(0, len(perm), bs):
            idx = perm[b : b + bs]
            if len(idx) < 2:  # InfoNCE exige au moins 2 paires
                continue
            i_t = torch.from_numpy(idx).to(device)
            r_t = torch.from_numpy(rng.integers(0, K, size=len(idx))).to(
                device
            )  # 1 légende / clip / batch
            loss = info_nce(model(X[i_t]), T[i_t, r_t], model.logit_scale)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            with torch.no_grad():
                model.logit_scale.clamp_(max=LOGIT_SCALE_MAX)
            losses.append(loss.item())
        m = evaluate(project(model, raw[va], device), cap[va])
        rec = {
            "epoch": epoch + 1,
            "loss": float(np.mean(losses)),
            "tau": float(torch.exp(-model.logit_scale).item()),
            **m,
        }
        history.append(rec)
        print(
            f"epoch {rec['epoch']:3d} | loss {rec['loss']:.4f} | tau {rec['tau']:.4f} | "
            f"a2t R@1 {m['a2t_R@1']:.3f} R@10 {m['a2t_R@10']:.3f} | rsum {m['rsum']:.3f}"
        )
        if m["rsum"] > best:
            best, bad = m["rsum"], 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= t["patience"]:
                print("arrêt anticipé")
                break
    model.load_state_dict(best_state)
    result = {"best_val_rsum": best, "history": history}
    if not overfit:
        result["test"] = evaluate(project(model, raw[te], device), cap[te])
        cfg.model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": best_state,
                "config": {
                    "in_dim": AST_DIM,
                    "hidden": t["hidden"],
                    "out_dim": cfg.emb_dim,
                    "dropout": t["dropout"],
                },
            },
            cfg.model_dir / "audio_proj.pt",
        )
        (cfg.model_dir / "audio_proj_log.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        save_embeddings(cfg.emb_dir / "audio_clotho.npy", project(model, raw, device), ids)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None)
    ap.add_argument("--overfit", type=int, default=0)
    a = ap.parse_args()
    train(load_config(a.config), a.overfit)
