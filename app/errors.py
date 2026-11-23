"""Gestion d'erreurs : format unique {"error": {"code", "message"}} (section 2.7.4)."""

from __future__ import annotations

from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

from mmsearch.retrieval.engine import CapabilityError, MediaError, NotFound, QueryError


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def _json_error(status: int, code: str, message: str):
    return jsonify({"error": {"code": code, "message": message}}), status


def register_error_handlers(app: Flask) -> None:
    app.register_error_handler(ApiError, lambda e: _json_error(e.status, e.code, e.message))
    app.register_error_handler(QueryError, lambda e: _json_error(400, "invalid_query", str(e)))
    app.register_error_handler(
        MediaError, lambda e: _json_error(415, "unsupported_media_type", str(e))
    )
    app.register_error_handler(
        CapabilityError, lambda e: _json_error(503, "modality_unavailable", str(e))
    )
    app.register_error_handler(NotFound, lambda e: _json_error(404, "not_found", f"inconnu : {e}"))
    app.register_error_handler(
        RequestEntityTooLarge,
        lambda e: _json_error(
            413,
            "file_too_large",
            f"fichier trop volumineux (max {app.config['MAX_CONTENT_LENGTH'] // 2**20} Mo)",
        ),
    )

    @app.errorhandler(HTTPException)
    def _http(e: HTTPException):
        if request.path.startswith(("/api/", "/media/")):
            name = e.name.lower().replace(" ", "_")
            return _json_error(e.code or 500, name, e.description or e.name)
        return e
