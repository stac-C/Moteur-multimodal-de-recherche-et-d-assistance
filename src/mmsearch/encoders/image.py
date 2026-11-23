"""Encodeur image : tour visuel de CLIP ViT-B/32 (patchs 32x32, 512 d), gelé."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from mmsearch.data.preprocess import load_image
from mmsearch.encoders.base import EMB_DIM, batched, l2_normalize


class ClipImageEncoder:
    modality = "image"

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str = "cpu",
        fp16: bool = False,
        workers: int = 8,
    ) -> None:
        from transformers import CLIPImageProcessor, CLIPModel

        self.device = device
        self.fp16 = fp16 and device.startswith("cuda")
        self.workers = workers
        self.processor = CLIPImageProcessor.from_pretrained(model_name)
        model = CLIPModel.from_pretrained(model_name).to(device).eval()
        self.model = model.half() if self.fp16 else model

    @torch.no_grad()
    def encode(self, inputs: Sequence["str | Path | object"], batch_size: int = 64) -> np.ndarray:
        """`inputs` : chemins ou objets PIL.Image. Le décodage disque est parallélisé (goulot réel)."""
        if len(inputs) == 0:
            return np.zeros((0, EMB_DIM), dtype=np.float32)
        out = []
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for chunk in batched(inputs, batch_size):
                images = list(
                    pool.map(
                        lambda x: load_image(x) if isinstance(x, (str, Path)) else x.convert("RGB"),
                        chunk,
                    )
                )
                pix = self.processor(images=images, return_tensors="pt")["pixel_values"].to(
                    self.device
                )
                if self.fp16:
                    pix = pix.half()
                out.append(self.model.get_image_features(pixel_values=pix).float().cpu().numpy())
        return l2_normalize(np.concatenate(out))
