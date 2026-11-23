"""GET /api/health."""

from __future__ import annotations

from flask import Blueprint, current_app, jsonify

import mmsearch

bp = Blueprint("health", __name__)


@bp.get("/api/health")
def health():
    engine = current_app.extensions["engine"]
    return jsonify(
        {
            "status": "ok",
            "capabilities": engine.capabilities(),
            "n_items": len(engine.items),
            "version": mmsearch.__version__,
        }
    )
