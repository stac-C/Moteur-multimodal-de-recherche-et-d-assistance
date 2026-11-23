"""VectorStore : index FAISS exact par produit scalaire (IndexFlatIP), section 2.3 / 2.8."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

import faiss
import numpy as np


class VectorStore:
    def __init__(self, dim: int = 512) -> None:
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.ids: list[str] = []
        self._pos: dict[str, int] = {}

    def __len__(self) -> int:
        return len(self.ids)

    def build(self, vectors: np.ndarray, ids: Sequence[str]) -> None:
        v = np.ascontiguousarray(vectors, dtype=np.float32)
        if v.ndim != 2 or v.shape[1] != self.dim:
            raise ValueError(f"forme attendue (n, {self.dim}), reçue {v.shape}")
        if len(ids) != v.shape[0] or len(set(ids)) != len(ids):
            raise ValueError("ids de longueur différente ou non uniques")
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(v)
        self.ids = list(ids)
        self._pos = {i: p for p, i in enumerate(self.ids)}

    def search(
        self, queries: np.ndarray, k: int, id_filter: Optional[set[str]] = None
    ) -> tuple[np.ndarray, list[list[str]]]:
        """Retourne (scores (nq, k'), ids par requête), k' = min(k, nombre d'items autorisés).

        id_filter : pré-filtrage exact par sélecteur FAISS (items autorisés uniquement).
        """
        q = np.ascontiguousarray(np.atleast_2d(queries), dtype=np.float32)
        if q.shape[1] != self.dim:
            raise ValueError(f"requête de dimension {q.shape[1]}, attendu {self.dim}")
        if id_filter is None:
            k_eff = min(k, len(self))
            scores, idx = self.index.search(q, k_eff)
        else:
            pos = np.array(
                sorted(self._pos[i] for i in id_filter if i in self._pos), dtype=np.int64
            )
            k_eff = min(k, len(pos))
            if k_eff == 0:
                return np.zeros((q.shape[0], 0), dtype=np.float32), [[] for _ in range(q.shape[0])]
            params = faiss.SearchParameters()
            params.sel = faiss.IDSelectorBatch(pos)
            scores, idx = self.index.search(q, k_eff, params=params)
        return scores, [[self.ids[j] for j in row] for row in idx]

    def __contains__(self, item_id: str) -> bool:
        return item_id in self._pos

    def vector(self, item_id: str) -> np.ndarray:
        """Vecteur stocké pour un id (reconstruction exacte, IndexFlat)."""
        return self.index.reconstruct(self._pos[item_id])

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(p))
        p.with_suffix(".ids.json").write_text(
            json.dumps(self.ids, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str) -> "VectorStore":
        p = Path(path)
        store = cls(dim=0)
        store.index = faiss.read_index(str(p))
        store.dim = store.index.d
        store.ids = json.loads(p.with_suffix(".ids.json").read_text(encoding="utf-8"))
        store._pos = {i: n for n, i in enumerate(store.ids)}
        return store
