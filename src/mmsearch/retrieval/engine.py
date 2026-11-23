"""SearchEngine : routage des requêtes texte / image / audio vers les index (section 2.3).

Routage (mode « fused » par défaut, mode « modal » pour les comparaisons mono-modales) :
  texte -> fused : IDX_FUSED                 | modal : IDX_IMG et IDX_AUD (listes fusionnées par score)
  image -> fused : IDX_TXT agrégé par item   | modal : IDX_IMG (image -> image)
  audio -> fused : IDX_FUSED (audio -> contenu) | modal : IDX_TXT agrégé par item (audio -> texte)
L'agrégation par item garde le meilleur score de ses légendes. Les encodeurs sont chargés à la
première utilisation (verrou) ; sans fabrique pour une modalité, elle est « indisponible » (503).
"""

from __future__ import annotations

import io
import os
import sqlite3
import tempfile
import threading
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from mmsearch.config import Config, resolve_device
from mmsearch.index.store import VectorStore
from mmsearch.retrieval.schemas import Hit, Query
from mmsearch.tasks.zero_shot_cls import class_prompts, classify_one

ALLOWED_FILTERS = {"modality", "dataset", "split", "label", "min_score"}
MAX_TEXT_CHARS = 500


class QueryError(ValueError):
    """Requête invalide (HTTP 400)."""


class MediaError(ValueError):
    """Média illisible (HTTP 415)."""


class CapabilityError(RuntimeError):
    """Modalité indisponible (HTTP 503)."""


class NotFound(KeyError):
    """Item inconnu (HTTP 404)."""


class SearchEngine:
    def __init__(
        self,
        cfg: Config,
        stores: dict[str, VectorStore],
        items: dict[str, dict],
        captions: dict[str, list[tuple[str, str]]],
        factories: Optional[dict[str, Callable[[], Any]]] = None,
    ) -> None:
        self.cfg, self.stores, self.items, self.captions = cfg, stores, items, captions
        self._factories = dict(factories or {})
        self._enc: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._cap_ids = {i: [c for c, _ in v] for i, v in captions.items()}
        self._cap_text = {c: t for v in captions.values() for c, t in v}
        self._cap_item = {c: i for i, v in captions.items() for c, _ in v}
        self._class_cache: dict[tuple, np.ndarray] = {}
        self.esc_classes = sorted({m["label"] for m in items.values() if m["dataset"] == "esc50"})

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_artifacts(
        cls,
        cfg: Config,
        load_models: bool = True,
        encoders: Optional[dict[str, Callable[[], Any]]] = None,
    ) -> "SearchEngine":
        stores = {
            n: VectorStore.load(str(cfg.index_dir / f"{n}.faiss"))
            for n in ("idx_img", "idx_aud", "idx_txt", "idx_fused")
        }
        con = sqlite3.connect(cfg.index_dir / "collection.db")
        con.row_factory = sqlite3.Row
        items = {r["item_id"]: dict(r) for r in con.execute("SELECT * FROM items")}
        captions: dict[str, list[tuple[str, str]]] = {}
        for r in con.execute(
            "SELECT caption_id, item_id, text FROM captions ORDER BY item_id, idx"
        ):
            captions.setdefault(r["item_id"], []).append((r["caption_id"], r["text"]))
        con.close()
        factories = encoders
        if factories is None and load_models:
            dev, m, a = resolve_device(cfg["device"]), cfg["models"], cfg["audio"]
            from mmsearch.encoders.image import ClipImageEncoder
            from mmsearch.encoders.text import ClipTextEncoder

            factories = {
                "text": lambda: ClipTextEncoder(m["clip"], dev),
                "image": lambda: ClipImageEncoder(m["clip"], dev),
            }
            ckpt = cfg.model_dir / "audio_proj.pt"
            if ckpt.exists():
                from mmsearch.encoders.audio import AudioEncoder

                factories["audio"] = lambda: AudioEncoder(
                    ckpt,
                    m["ast"],
                    dev,
                    sample_rate=a["sample_rate"],
                    window_s=a["window_s"],
                    min_tail_s=a["min_tail_s"],
                    max_seconds=a["max_upload_s"],
                )
        return cls(cfg, stores, items, captions, factories)

    def capabilities(self) -> dict[str, bool]:
        return {k: k in self._factories for k in ("text", "image", "audio")}

    def stats(self) -> dict[str, int]:
        return {
            "n_items": len(self.items),
            "n_captions": len(self._cap_item),
            **{n: len(s) for n, s in self.stores.items()},
        }

    def _encoder(self, kind: str):
        if kind not in self._factories:
            raise CapabilityError(f"modalité « {kind} » indisponible")
        with self._lock:
            if kind not in self._enc:
                self._enc[kind] = self._factories[kind]()
        return self._enc[kind]

    # ------------------------------------------------------------------ encodage de la requête
    def _embed_audio(self, payload: str | bytes | Path, suffix: str = "") -> np.ndarray:
        enc = self._encoder("audio")
        tmp = None
        try:
            if isinstance(payload, (bytes, bytearray)):
                fd, tmp = tempfile.mkstemp(suffix=suffix or ".audio")
                with os.fdopen(fd, "wb") as f:
                    f.write(payload)
                path = tmp
            else:
                path = str(payload)
            return enc.encode([path])[0]
        except (CapabilityError, QueryError):
            raise
        except Exception as e:  # décodage audio impossible
            raise MediaError(f"audio illisible : {e}") from e
        finally:
            if tmp and os.path.exists(tmp):
                os.remove(tmp)

    def _embed_image(self, payload: str | bytes | Path) -> np.ndarray:
        enc = self._encoder("image")
        try:
            if isinstance(payload, (bytes, bytearray)):
                from PIL import Image

                payload = Image.open(io.BytesIO(payload)).convert("RGB")
            elif isinstance(payload, Path):
                payload = str(payload)
            return enc.encode([payload])[0]
        except (CapabilityError, QueryError):
            raise
        except Exception as e:
            raise MediaError(f"image illisible : {e}") from e

    def _embed(self, q: Query) -> np.ndarray:
        if q.kind == "text":
            if not isinstance(q.payload, str) or not q.payload.strip():
                raise QueryError("texte vide")
            if len(q.payload) > MAX_TEXT_CHARS:
                raise QueryError(f"texte trop long (> {MAX_TEXT_CHARS} caractères)")
            return self._encoder("text").encode([q.payload])[0]
        if q.kind == "image":
            return self._embed_image(q.payload)
        return self._embed_audio(q.payload, q.suffix)

    # ------------------------------------------------------------------ validation et filtres
    def _validate(self, q: Query) -> None:
        if q.kind not in ("text", "image", "audio"):
            raise QueryError(f"kind invalide : {q.kind}")
        if q.mode not in ("fused", "modal"):
            raise QueryError(f"mode invalide : {q.mode}")
        if not isinstance(q.k, int) or not 1 <= q.k <= self.cfg["search"]["max_k"]:
            raise QueryError(f"k doit être entre 1 et {self.cfg['search']['max_k']}")
        bad = set(q.filters) - ALLOWED_FILTERS
        if bad:
            raise QueryError(f"filtres inconnus : {sorted(bad)}")
        if q.filters.get("modality") not in (None, "image", "audio"):
            raise QueryError("filters.modality doit valoir image ou audio")
        if "min_score" in q.filters and not isinstance(q.filters["min_score"], (int, float)):
            raise QueryError("filters.min_score doit être un nombre")

    def _allowed_items(self, filters: dict[str, Any]) -> Optional[set[str]]:
        keys = [k for k in ("modality", "dataset", "split", "label") if filters.get(k)]
        if not keys:
            return None
        return {i for i, m in self.items.items() if all(m[k] == filters[k] for k in keys)}

    # ------------------------------------------------------------------ recherche
    def _search_items(self, name: str, z: np.ndarray, k: int, allowed: Optional[set[str]]):
        if allowed is not None and not allowed:
            return []
        sc, ids = self.stores[name].search(z, k, id_filter=allowed)
        return [(i, float(s)) for i, s in zip(ids[0], sc[0])]

    def _search_captions(self, z: np.ndarray, k: int, allowed: Optional[set[str]]):
        """Recherche dans IDX_TXT puis agrégation par item (meilleure légende)."""
        store = self.stores["idx_txt"]
        cap_filter = None
        if allowed is not None:
            cap_filter = {c for i in allowed for c in self._cap_ids.get(i, ())}
            if not cap_filter:
                return []
        total = len(store) if cap_filter is None else len(cap_filter)
        k_try = min(total, self.cfg["search"]["overfetch"] * k)
        while True:
            sc, ids = store.search(z, k_try, id_filter=cap_filter)
            best: dict[str, tuple[float, str]] = {}
            for cid, s in zip(ids[0], sc[0]):
                best.setdefault(self._cap_item[cid], (float(s), cid))
            if len(best) >= k or k_try >= total:
                break
            k_try = min(total, 2 * k_try)
        return [(i, s, self._cap_text[c]) for i, (s, c) in list(best.items())[:k]]

    def _modal_scores(self, z: np.ndarray, item_id: str) -> dict[str, float]:
        out: dict[str, float] = {}
        m = self.items[item_id]["modality"]
        store = self.stores["idx_img" if m == "image" else "idx_aud"]
        if item_id in store:
            out[m] = float(z @ store.vector(item_id))
        cids = self._cap_ids.get(item_id)
        if cids:
            c = np.mean([self.stores["idx_txt"].vector(i) for i in cids], axis=0)
            out["caption"] = float(z @ (c / np.linalg.norm(c)))
        return out

    def _best_caption(self, z: np.ndarray, item_id: str) -> Optional[str]:
        cids = self._cap_ids.get(item_id)
        if not cids:
            return None
        sims = [float(z @ self.stores["idx_txt"].vector(c)) for c in cids]
        return self._cap_text[cids[int(np.argmax(sims))]]

    def search(self, q: Query) -> list[Hit]:
        self._validate(q)
        z = self._embed(q)
        allowed = self._allowed_items(q.filters)
        k, mod = q.k, q.filters.get("modality")
        rows: list[tuple[str, float, Optional[str]]]
        if (q.kind, q.mode) in (("text", "fused"), ("audio", "fused")):
            rows = [(i, s, None) for i, s in self._search_items("idx_fused", z, k, allowed)]
        elif (q.kind, q.mode) == ("text", "modal"):
            rows = []
            for name, m in (("idx_img", "image"), ("idx_aud", "audio")):
                if mod in (None, m):
                    rows += [(i, s, None) for i, s in self._search_items(name, z, k, allowed)]
            rows = sorted(rows, key=lambda r: -r[1])[:k]
        elif (q.kind, q.mode) == ("image", "modal"):
            rows = [(i, s, None) for i, s in self._search_items("idx_img", z, k, allowed)]
        else:  # image fused, audio modal : légendes agrégées
            rows = self._search_captions(z, k, allowed)
        if "min_score" in q.filters:
            rows = [r for r in rows if r[1] >= q.filters["min_score"]]
        hits = []
        for rank, (item_id, score, cap) in enumerate(rows, 1):
            meta = self.items[item_id]
            hits.append(
                Hit(
                    rank=rank,
                    item_id=item_id,
                    modality=meta["modality"],
                    score=score,
                    scores_by_modality=self._modal_scores(z, item_id),
                    caption=cap or self._best_caption(z, item_id),
                    media_url=f"/media/{item_id}",
                    dataset=meta["dataset"],
                    label=meta["label"],
                )
            )
        return hits

    # ------------------------------------------------------------------ autres services
    def get_item(self, item_id: str) -> dict[str, Any]:
        if item_id not in self.items:
            raise NotFound(item_id)
        meta = self.items[item_id]
        return {
            **meta,
            "item_id": item_id,
            "media_url": f"/media/{item_id}",
            "captions": [{"caption_id": c, "text": t} for c, t in self.captions.get(item_id, [])],
        }

    def media_path(self, item_id: str) -> Path:
        """Chemin du fichier média ; refuse tout chemin hors de data/ (anti-traversée)."""
        if item_id not in self.items:
            raise NotFound(item_id)
        p = (self.cfg.root / self.items[item_id]["path"]).resolve()
        base = (self.cfg.root / self.cfg["paths"]["data"]).resolve()
        if base not in p.parents:
            raise NotFound(item_id)
        return p

    def _class_embeddings(self, classes: tuple[str, ...]) -> np.ndarray:
        if classes not in self._class_cache:
            self._class_cache[classes] = self._encoder("text").encode(class_prompts(list(classes)))
        return self._class_cache[classes]

    def classify_audio(
        self,
        payload: str | bytes | Path,
        classes: Optional[list[str]] = None,
        suffix: str = "",
        topk: int = 5,
    ) -> list[tuple[str, float]]:
        """Classification zero-shot (section 1.6) d'un clip : distribution softmax(100 * cos)."""
        cls = tuple(classes) if classes else tuple(self.esc_classes)
        if len(cls) < 2:
            raise QueryError("au moins 2 classes requises")
        z = self._embed_audio(payload, suffix)
        return classify_one(z, self._class_embeddings(cls), list(cls), topk=topk)
