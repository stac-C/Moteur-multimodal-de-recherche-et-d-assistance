"""Objets d'échange du moteur de recherche (contrat 2.7.3, avec trois champs optionnels ajoutés)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional


@dataclass(frozen=True)
class Query:
    kind: Literal["text", "image", "audio"]
    payload: str | bytes | Path  # texte, ou contenu / chemin du média
    k: int = 10  # 1 <= k <= 50
    filters: dict[str, Any] = field(default_factory=dict)
    mode: Literal["fused", "modal"] = "fused"  # ajout : index fusionné ou index mono-modaux
    suffix: str = ""  # ajout : extension du fichier téléversé (ex. ".wav")


@dataclass(frozen=True)
class Hit:
    rank: int
    item_id: str
    modality: str
    score: float
    scores_by_modality: dict[str, float]
    caption: Optional[str]
    media_url: str
    dataset: str = ""  # ajout
    label: str = ""  # ajout (classe ESC-50)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["score"] = round(d["score"], 4)
        d["scores_by_modality"] = {k: round(v, 4) for k, v in d["scores_by_modality"].items()}
        return d
