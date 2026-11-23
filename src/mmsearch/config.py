"""Chargement de config.yaml, choix du device, graine globale."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Config:
    data: dict[str, Any]
    root: Path

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    @property
    def seed(self) -> int:
        return int(self.data["seed"])

    @property
    def emb_dim(self) -> int:
        return int(self.data["emb_dim"])

    @property
    def artifacts(self) -> Path:
        return self.root / self.data["paths"]["artifacts"]

    @property
    def emb_dir(self) -> Path:
        return self.artifacts / "embeddings"

    @property
    def model_dir(self) -> Path:
        return self.artifacts / "models"

    @property
    def index_dir(self) -> Path:
        return self.artifacts / "index"

    @property
    def processed(self) -> Path:
        return self.root / self.data["paths"]["data"] / "processed"

    @property
    def raw(self) -> Path:
        return self.root / self.data["paths"]["data"] / "raw"


def load_config(path: str | Path | None = None) -> Config:
    p = Path(path) if path else ROOT / "config.yaml"
    with open(p, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Config(data=data, root=p.resolve().parent)


def resolve_device(name: str = "auto") -> str:
    if name != "auto":
        return name
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
