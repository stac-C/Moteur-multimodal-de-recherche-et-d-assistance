import io
import json

import pytest

from app import create_app
from tests.conftest import png_bytes


@pytest.fixture()
def client(cfg, engine):
    app = create_app(cfg, engine)
    app.config["TESTING"] = True
    return app.test_client()


def post(client, **form):
    return client.post("/api/search", data=form, content_type="multipart/form-data")


def test_health_and_pages(client):
    j = client.get("/api/health").get_json()
    assert j["status"] == "ok" and j["capabilities"] == {"text": True, "image": True, "audio": True}
    assert j["n_items"] == 130
    for url in ("/", "/embeddings"):
        r = client.get(url)
        assert r.status_code == 200 and b"<!doctype html>" in r.data
    assert client.get("/static/js/app.js").status_code == 200


def test_search_text_contract(client):
    r = post(
        client, kind="text", q="fake caption 2 1", k="3", filters=json.dumps({"modality": "image"})
    )
    j = r.get_json()
    assert r.status_code == 200 and j["query"]["k"] == 3 and len(j["hits"]) == 3
    h = j["hits"][0]
    assert {
        "rank",
        "item_id",
        "modality",
        "score",
        "scores_by_modality",
        "caption",
        "media_url",
    } <= set(h)
    assert h["item_id"] == "img_000002" and j["query"]["latency_ms"] >= 0


def test_search_image_and_audio_uploads(client):
    r = post(client, kind="image", k="3", file=(io.BytesIO(png_bytes(9)), "q.png"))
    assert r.status_code == 200 and r.get_json()["hits"][0]["item_id"] == "img_000009"
    r = post(client, kind="audio", k="3", file=(io.BytesIO(b"4"), "q.wav"))
    assert r.status_code == 200 and r.get_json()["hits"][0]["item_id"] == "aud_000004"


@pytest.mark.parametrize(
    "form,status,code",
    [
        ({"kind": "video"}, 400, "invalid_query"),
        ({"kind": "text", "q": "a", "k": "0"}, 400, "invalid_query"),
        ({"kind": "text", "q": "a", "k": "abc"}, 400, "invalid_query"),
        ({"kind": "text", "q": "a", "filters": "[1]"}, 400, "invalid_query"),
        ({"kind": "text", "q": "a", "filters": "{pas du json"}, 400, "invalid_query"),
        ({"kind": "text", "q": ""}, 400, "invalid_query"),
        ({"kind": "image"}, 400, "invalid_query"),
    ],
)
def test_search_errors(client, form, status, code):
    r = post(client, **form)
    assert r.status_code == status and r.get_json()["error"]["code"] == code


def test_upload_validation(client):
    r = post(client, kind="image", file=(io.BytesIO(b"x"), "q.gif"))
    assert r.status_code == 415 and r.get_json()["error"]["code"] == "unsupported_media_type"
    assert (
        post(client, kind="image", file=(io.BytesIO(b"pas une image"), "q.png")).status_code == 415
    )
    assert post(client, kind="audio", file=(io.BytesIO(b"abc"), "q.wav")).status_code == 415
    assert post(client, kind="audio", file=(io.BytesIO(b""), "q.wav")).status_code == 400


def test_upload_too_large(cfg, engine):
    app = create_app(cfg, engine)
    app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024
    r = app.test_client().post(
        "/api/search",
        content_type="multipart/form-data",
        data={"kind": "audio", "file": (io.BytesIO(b"0" * (2 * 1024 * 1024)), "q.wav")},
    )
    assert r.status_code == 413 and r.get_json()["error"]["code"] == "file_too_large"


def test_item_media_classify(client):
    j = client.get("/api/item/aud_000003").get_json()
    assert j["modality"] == "audio" and len(j["captions"]) == 5
    assert client.get("/api/item/zzz").status_code == 404
    assert client.get("/media/img_000001").status_code == 404  # fichier factice absent
    assert client.get("/api/inconnue").get_json()["error"]["code"] == "not_found"
    r = client.post(
        "/api/classify",
        data={"file": (io.BytesIO(b"3"), "c.wav")},
        content_type="multipart/form-data",
    )
    j = r.get_json()
    assert (
        r.status_code == 200
        and len(j["classes"]) == 5
        and abs(sum(c["prob"] for c in j["classes"])) <= 1.0001
    )
    bad = client.post(
        "/api/classify",
        data={"file": (io.BytesIO(b"3"), "c.wav"), "classes": "[1]"},
        content_type="multipart/form-data",
    )
    assert bad.status_code == 400


def test_embeddings_api(client):
    j = client.get("/api/embeddings?method=pca&limit=100").get_json()
    assert j["n_points"] == 100 and len(j["explained_variance"]) == 2
    assert {"id", "x", "y", "modality", "dataset", "label"} <= set(j["points"][0])
    only = client.get("/api/embeddings?modality=audio&dataset=esc50").get_json()
    assert only["n_points"] == 30 and {p["label"] != "" for p in only["points"]} == {True}
    assert client.get("/api/embeddings?method=xyz").status_code == 400
    assert client.get("/api/embeddings?limit=99999").status_code == 400
    assert client.get("/api/embeddings?method=tsne").status_code == 503  # non précalculée
