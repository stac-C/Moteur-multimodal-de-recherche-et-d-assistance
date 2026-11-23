"""POST /api/classify : classification audio zero-shot (tâche secondaire)."""

from __future__ import annotations

import json
import time

from flask import Blueprint, current_app, jsonify, request

from app.errors import ApiError
from app.routes.search import AUDIO_EXT, read_upload

bp = Blueprint("audio", __name__)


@bp.post("/api/classify")
def api_classify():
    engine = current_app.extensions["engine"]
    data, ext = read_upload("file", AUDIO_EXT)
    classes = None
    if request.form.get("classes"):
        try:
            classes = json.loads(request.form["classes"])
            assert isinstance(classes, list) and all(isinstance(c, str) for c in classes)
        except (ValueError, AssertionError) as e:
            raise ApiError(
                400, "invalid_query", "classes doit être une liste JSON de chaînes"
            ) from e
    t0 = time.perf_counter()
    top = engine.classify_audio(data, classes, suffix=ext)
    return jsonify(
        {
            "latency_ms": round((time.perf_counter() - t0) * 1000),
            "classes": [{"label": c, "prob": round(p, 4)} for c, p in top],
        }
    )
