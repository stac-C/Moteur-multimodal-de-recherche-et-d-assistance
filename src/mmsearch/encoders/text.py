"""Encodeurs texte : CLIP (espace commun, 512 d) et SBERT (baseline mono-modale, 384 d)."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from mmsearch.data.preprocess import clean_text
from mmsearch.encoders.base import EMB_DIM, batched, l2_normalize


class ClipTextEncoder:
    """Tour texte de CLIP ViT-B/32, gelé. Troncature à 77 tokens (E1 : jeton [EOS])."""

    modality = "text"

    def __init__(
        self, model_name: str = "openai/clip-vit-base-patch32", device: str = "cpu"
    ) -> None:
        from transformers import CLIPModel, CLIPTokenizerFast

        self.device = device
        self.tokenizer = CLIPTokenizerFast.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(device).eval()

    @torch.no_grad()
    def encode(self, texts: Sequence[str], batch_size: int = 128) -> np.ndarray:
        if len(texts) == 0:
            return np.zeros((0, EMB_DIM), dtype=np.float32)
        out = []
        for chunk in batched([clean_text(t) for t in texts], batch_size):
            tok = self.tokenizer(
                list(chunk), padding=True, truncation=True, max_length=77, return_tensors="pt"
            ).to(self.device)
            out.append(self.model.get_text_features(**tok).float().cpu().numpy())
        return l2_normalize(np.concatenate(out))


class SbertTextEncoder:
    """Baseline mono-texte : all-MiniLM-L6-v2 (moyenne masquée, E1), 384 d. Hors espace commun."""

    modality = "text"

    def __init__(
        self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu"
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: Sequence[str], batch_size: int = 128) -> np.ndarray:
        emb = self.model.encode(
            [clean_text(t) for t in texts],
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.ascontiguousarray(emb, dtype=np.float32)
