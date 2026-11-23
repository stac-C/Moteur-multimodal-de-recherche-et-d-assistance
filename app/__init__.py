"""Application Flask : fabrique create_app() (section 2.6). Lancer : flask --app app run."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from flask import Flask
from dotenv import load_dotenv

from app.errors import register_error_handlers
from mmsearch.config import Config, load_config
from mmsearch.retrieval.engine import SearchEngine


MAX_UPLOAD_BYTES = 20 * 1024 * 1024

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def create_app(cfg: Optional[Config] = None, engine: Optional[SearchEngine] = None) -> Flask:
    cfg = cfg or load_config()
    here = Path(__file__).parent
    app = Flask(
        __name__, template_folder=str(here / "templates"), static_folder=str(here / "static")
    )
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
    app.json.ensure_ascii = False
    app.extensions["cfg"] = cfg
    # MMSEARCH_LOAD_MODELS=0 : démarre sans modèles (pages, filtres, projections) ; les onglets sont désactivés
    load = os.environ.get("MMSEARCH_LOAD_MODELS", "1") != "0"
    app.extensions["engine"] = engine or SearchEngine.from_artifacts(cfg, load_models=load)

    from app.routes import audio, health, items, search, viz

    for module in (search, items, audio, viz, health):
        app.register_blueprint(module.bp)
    register_error_handlers(app)
    return app


# Entry-point WSGI utilisé par les serveurs compatibles Flask/Vercel.
app = create_app()
