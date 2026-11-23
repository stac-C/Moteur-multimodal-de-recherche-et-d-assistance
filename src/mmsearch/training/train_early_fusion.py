"""Entraîne la fusion précoce (E6) sur Flickr8k : requête = une légende retirée, positif = item fusionné.

Leave-one-out : le contexte d'un item exclut la légende qui sert de requête (pas de fuite).
Initialisation = fusion tardive (w_mod, w_cap) ; on ne garde le modèle que s'il bat cette référence
sur la validation (critère rsum = R@1 + R@5 + R@10).
Sorties : models/early_fusion.pt, models/early_fusion_log.json.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from mmsearch.config import load_config, resolve_device, set_seed
from mmsearch.data.loaders import captions_tensor, load_captions, load_manifest
from mmsearch.embeddings_io import load_embeddings
from mmsearch.fusion.early import EarlyFusion
from mmsearch.fusion.late import caption_context, evaluate_text_to_item
from mmsearch.training.losses import LOGIT_SCALE_MAX, info_nce

K = 5


def main(cfg, w_mod: float) -> dict:
    t = cfg["train_early"]
    set_seed(cfg.seed)
    device = resolve_device(cfg["device"])
    man, caps = load_manifest(cfg), load_captions(cfg)
    img, ids = load_embeddings(cfg.emb_dir / "clip_img_flickr8k.npy")
    cap_emb, cap_ids = load_embeddings(cfg.emb_dir / "clip_txt_flickr8k.npy")
    cap = captions_tensor(ids, caps, cap_emb, cap_ids, K)
    split_of = man.set_index("item_id")["split"].to_dict()
    tr, va, te = (
        np.array([i for i, it in enumerate(ids) if split_of[it] == s])
        for s in ("train", "val", "test")
    )

    model = EarlyFusion(cfg.emb_dim, w_mod, 1.0 - w_mod).to(device)
    ref = evaluate_text_to_item(lambda a, b: model.fuse_numpy(a, b, device), img[va], cap[va])
    print(f"référence (init = fusion tardive w_mod={w_mod}) : rsum val = {ref['rsum']:.3f}")
    best, best_state, bad, history = ref["rsum"], None, 0, [{"epoch": 0, **ref}]

    _, ctx_loo = caption_context(cap)  # (n, K, d), contexte sans la légende r
    X, C, Q = (torch.from_numpy(a).to(device) for a in (img, ctx_loo, cap))
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    rng, bs = np.random.default_rng(cfg.seed), t["batch_size"]
    for epoch in range(t["epochs"]):
        model.train()
        perm, losses = rng.permutation(tr), []
        for b in range(0, len(perm), bs):
            idx = perm[b : b + bs]
            if len(idx) < 2:
                continue
            i_t = torch.from_numpy(idx).to(device)
            r_t = torch.from_numpy(rng.integers(0, K, size=len(idx))).to(device)
            loss = info_nce(Q[i_t, r_t], model(X[i_t], C[i_t, r_t]), model.logit_scale)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            with torch.no_grad():
                model.logit_scale.clamp_(max=LOGIT_SCALE_MAX)
            losses.append(loss.item())
        m = evaluate_text_to_item(lambda a, b: model.fuse_numpy(a, b, device), img[va], cap[va])
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), **m})
        print(
            f"epoch {epoch + 1:3d} | loss {history[-1]['loss']:.4f} | R@1 {m['R@1']:.3f} | rsum {m['rsum']:.3f}"
        )
        if m["rsum"] > best:
            best, bad = m["rsum"], 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= t["patience"]:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    result = {
        "init_val": ref,
        "best_val_rsum": best,
        "improved_over_late": best_state is not None,
        "test": evaluate_text_to_item(
            lambda a, b: model.fuse_numpy(a, b, device), img[te], cap[te]
        ),
        "history": history,
    }
    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "w_mod_init": w_mod}, cfg.model_dir / "early_fusion.pt"
    )
    (cfg.model_dir / "early_fusion_log.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None)
    ap.add_argument(
        "--w-mod", type=float, default=None, help="poids d'initialisation (défaut : config)"
    )
    a = ap.parse_args()
    c = load_config(a.config)
    main(c, a.w_mod if a.w_mod is not None else c["fusion"]["weights_image"]["image"])
