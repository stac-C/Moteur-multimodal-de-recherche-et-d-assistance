"""POST /api/search et page d'accueil."""

from __future__ import annotations

import io
import json
import time
from pathlib import Path

from flask import Blueprint, current_app, jsonify, render_template, request

from app.errors import ApiError
from mmsearch.retrieval.schemas import Query

bp = Blueprint("search", __name__)
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}
AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg"}


def read_upload(field: str, allowed: set[str]) -> tuple[bytes, str]:
    """Lit et valide un fichier téléversé (extension autorisée, non vide). Retourne (octets, extension)."""
    f = request.files.get(field)
    if f is None or not f.filename:
        raise ApiError(400, "invalid_query", f"fichier « {field} » requis")
    ext = Path(f.filename).suffix.lower()
    if ext not in allowed:
        raise ApiError(
            415,
            "unsupported_media_type",
            f"extension {ext or '(aucune)'} non supportée ; autorisées : {sorted(allowed)}",
        )
    data = f.read()
    if not data:
        raise ApiError(400, "invalid_query", "fichier vide")
    return data, ext


def parse_search_request(cfg) -> Query:
    kind = request.form.get("kind", "")
    if kind not in ("text", "image", "audio"):
        raise ApiError(400, "invalid_query", "kind doit valoir text, image ou audio")
    try:
        k = int(request.form.get("k", cfg["search"]["default_k"]))
        filters = json.loads(request.form.get("filters") or "{}")
    except ValueError as e:  # int() et json.JSONDecodeError héritent de ValueError
        raise ApiError(400, "invalid_query", f"paramètre mal formé : {e}") from e
    if not isinstance(filters, dict):
        raise ApiError(400, "invalid_query", "filters doit être un objet JSON")
    mode = request.form.get("mode", "fused")
    if kind == "text":
        return Query("text", request.form.get("q", ""), k, filters, mode)
    if kind == "image":
        data, ext = read_upload("file", IMAGE_EXT)
        try:
            from PIL import Image

            Image.open(io.BytesIO(data)).verify()
        except Exception as e:
            raise ApiError(415, "unsupported_media_type", "image illisible") from e
        return Query("image", data, k, filters, mode, ext)
    data, ext = read_upload("file", AUDIO_EXT)
    return Query("audio", data, k, filters, mode, ext)


@bp.get("/")
def index():
    engine = current_app.extensions["engine"]
    caps = {**engine.capabilities(), "classify": bool(engine.esc_classes)}
    return render_template("index.html", caps=caps)


@bp.post("/api/search")
def api_search():
    engine, cfg = current_app.extensions["engine"], current_app.extensions["cfg"]
    q = parse_search_request(cfg)
    t0 = time.perf_counter()
    hits = engine.search(q)
    ms = round((time.perf_counter() - t0) * 1000)
    return jsonify(
        {
            "query": {
                "kind": q.kind,
                "mode": q.mode,
                "k": q.k,
                "latency_ms": ms,
                "n_hits": len(hits),
            },
            "hits": [h.to_dict() for h in hits],
        }
    )
