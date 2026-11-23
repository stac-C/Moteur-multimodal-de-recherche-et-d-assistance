"""GET /api/item/<id> et GET /media/<id>."""

from __future__ import annotations

from flask import Blueprint, current_app, jsonify, send_file

from mmsearch.retrieval.engine import NotFound

bp = Blueprint("items", __name__)


@bp.get("/api/item/<item_id>")
def api_item(item_id: str):
    return jsonify(current_app.extensions["engine"].get_item(item_id))


@bp.get("/media/<item_id>")
def media(item_id: str):
    path = current_app.extensions["engine"].media_path(item_id)
    if not path.is_file():
        raise NotFound(item_id)
    return send_file(path, conditional=True)
