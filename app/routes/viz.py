"""Page de visualisation des embeddings et GET /api/embeddings."""

from __future__ import annotations

import numpy as np
from flask import Blueprint, current_app, jsonify, render_template, request

from app.errors import ApiError
from mmsearch.viz.projections import METHODS, load_or_compute

bp = Blueprint("viz", __name__)
MAX_POINTS = 2000


@bp.get("/embeddings")
def embeddings_page():
    return render_template("embeddings.html")


@bp.get("/api/embeddings")
def api_embeddings():
    cfg = current_app.extensions["cfg"]
    method = request.args.get("method", "pca")
    if method not in METHODS:
        raise ApiError(400, "invalid_query", f"method doit valoir {', '.join(METHODS)}")
    try:
        limit = int(request.args.get("limit", MAX_POINTS))
    except ValueError as e:
        raise ApiError(400, "invalid_query", "limit doit être un entier") from e
    if not 1 <= limit <= MAX_POINTS:
        raise ApiError(400, "invalid_query", f"limit doit être entre 1 et {MAX_POINTS}")
    try:
        proj = load_or_compute(cfg, method)
    except FileNotFoundError as e:
        raise ApiError(503, "projection_unavailable", str(e)) from e
    ds, mod = request.args.get("dataset"), request.args.get("modality")
    pts = [
        p
        for p in proj["points"]
        if ds in (None, "", p["dataset"]) and mod in (None, "", p["modality"])
    ]
    if len(pts) > limit:
        keep = np.sort(np.random.default_rng(cfg.seed).choice(len(pts), limit, replace=False))
        pts = [pts[i] for i in keep]
    return jsonify(
        {
            "method": method,
            "n_points": len(pts),
            "explained_variance": proj.get("explained_variance"),
            "points": pts,
        }
    )
