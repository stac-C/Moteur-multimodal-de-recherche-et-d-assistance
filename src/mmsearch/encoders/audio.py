"""Encodeur audio : AST gelé (768 d) + tête de projection MLP entraînée (512 d, équation E3).

Le log-Mel (E2) est calculé par l'extracteur d'AST : fbank de 128 bandes Mel, fenêtre de 25 ms,
pas de 10 ms (N = 400, H = 160 à 16 kHz), 1 024 trames pour 10 s.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmsearch.data.preprocess import load_audio, split_windows
from mmsearch.encoders.base import EMB_DIM, batched, l2_normalize

AST_DIM = 768


class AudioProjection(nn.Module):
    """u = W2 GELU(W1 h + b1) + b2 puis z = u / ||u|| (E3).

    Paramètres de la tête : 768*1024 + 1024 + 1024*512 + 512 = 1 312 256 ;
    `logit_scale` (= ln(1/tau), appris, E5) ajoute 1 paramètre utilisé seulement à l'entraînement.
    """

    def __init__(
        self,
        in_dim: int = AST_DIM,
        hidden: int = 1024,
        out_dim: int = EMB_DIM,
        dropout: float = 0.1,
        init_tau: float = 0.07,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, out_dim)
        )
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / init_tau)))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(h), dim=-1)


class AstBackbone:
    """AST pré-entraîné AudioSet, gelé. Sortie : pooler_output (moyenne CLS + distillation), 768 d."""

    def __init__(
        self,
        model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
        device: str = "cpu",
        sample_rate: int = 16000,
        window_s: float = 10.0,
        min_tail_s: float = 1.0,
        max_seconds: float | None = None,
    ) -> None:
        from transformers import ASTFeatureExtractor, ASTModel

        self.device = device
        self.sr, self.window_s, self.min_tail_s, self.max_seconds = (
            sample_rate,
            window_s,
            min_tail_s,
            max_seconds,
        )
        self.extractor = ASTFeatureExtractor.from_pretrained(model_name)
        self.model = ASTModel.from_pretrained(model_name).to(device).eval()

    @torch.no_grad()
    def encode_raw(self, paths: Sequence["str | Path"], batch_size: int = 8) -> np.ndarray:
        """Embeddings bruts (n, 768), non normalisés : moyenne des fenêtres de 10 s de chaque clip."""
        if len(paths) == 0:
            return np.zeros((0, AST_DIM), dtype=np.float32)
        result = []
        for chunk in batched(paths, batch_size):
            windows, owner = [], []
            for i, p in enumerate(chunk):
                wave = load_audio(p, self.sr, self.max_seconds)
                for w in split_windows(wave, self.sr, self.window_s, self.min_tail_s):
                    windows.append(w)
                    owner.append(i)
            feats = self.extractor(windows, sampling_rate=self.sr, return_tensors="pt")[
                "input_values"
            ]
            pooled = (
                self.model(input_values=feats.to(self.device)).pooler_output.float().cpu().numpy()
            )
            acc = np.zeros((len(chunk), AST_DIM), dtype=np.float64)
            cnt = np.zeros(len(chunk))
            np.add.at(acc, owner, pooled)
            np.add.at(cnt, owner, 1.0)
            result.append((acc / cnt[:, None]).astype(np.float32))
        return np.concatenate(result)


def load_projection(ckpt_path: str | Path, device: str = "cpu") -> AudioProjection:
    ck = torch.load(ckpt_path, map_location=device, weights_only=True)
    cfg = ck["config"]
    model = AudioProjection(cfg["in_dim"], cfg["hidden"], cfg["out_dim"], cfg["dropout"])
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval()


@torch.no_grad()
def project_raw(
    raw: np.ndarray, ckpt_path: str | Path, device: str = "cpu", batch_size: int = 1024
) -> np.ndarray:
    """Applique la tête entraînée à des embeddings AST bruts : (n, 768) -> (n, 512) normalisés."""
    model = load_projection(ckpt_path, device)
    out = [model(torch.from_numpy(c).to(device)).cpu().numpy() for c in batched(raw, batch_size)]
    return l2_normalize(np.concatenate(out)) if out else np.zeros((0, EMB_DIM), dtype=np.float32)


class AudioEncoder:
    """AST gelé + tête entraînée : chemins audio -> (n, 512) normalisés. Contrat Encoder (2.7.3)."""

    modality = "audio"

    def __init__(
        self,
        ckpt_path: str | Path,
        model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
        device: str = "cpu",
        **kw,
    ) -> None:
        self.backbone = AstBackbone(model_name, device, **kw)
        self.projection = load_projection(ckpt_path, device)
        self.device = device

    @torch.no_grad()
    def encode(self, paths: Sequence["str | Path"], batch_size: int = 8) -> np.ndarray:
        raw = self.backbone.encode_raw(paths, batch_size)
        z = self.projection(torch.from_numpy(raw).to(self.device)).cpu().numpy()
        return l2_normalize(z)
