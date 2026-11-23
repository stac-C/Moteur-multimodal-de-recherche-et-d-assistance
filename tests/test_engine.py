import numpy as np
import pytest

from mmsearch.retrieval.engine import (
    CapabilityError,
    MediaError,
    NotFound,
    QueryError,
    SearchEngine,
)
from mmsearch.retrieval.schemas import Query
from tests.conftest import png_bytes


def test_text_fused_returns_sorted_hits_with_owner_first(engine):
    hits = engine.search(Query("text", "fake caption 5 0", k=10))
    assert len(hits) == 10 and [h.rank for h in hits] == list(range(1, 11))
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
    assert hits[0].item_id == "img_000005"  # la légende est dans l'item fusionné
    assert set(hits[0].scores_by_modality) == {"image", "caption"}
    assert hits[0].media_url == "/media/img_000005" and hits[0].caption == "fake caption 5 0"


def test_text_modal_mono_image_retrieval(engine):
    hits_ok = 0
    for i in range(20):
        hits = engine.search(
            Query("text", f"fake caption {i} 0", k=10, mode="modal", filters={"modality": "image"})
        )
        hits_ok += f"img_{i:06d}" in [h.item_id for h in hits]
    assert hits_ok >= 16


def test_filters(engine):
    q = lambda **f: engine.search(Query("text", "fake sound 3 1", k=20, filters=f))  # noqa: E731
    assert {h.modality for h in q(modality="audio")} == {"audio"}
    assert {h.dataset for h in q(dataset="flickr8k")} == {"flickr8k"}
    test_hits = q(split="test")
    assert test_hits and {engine.items[h.item_id]["split"] for h in test_hits} == {"test"}
    assert q(dataset="inexistant") == []
    assert all(h.score >= 0.5 for h in q(min_score=0.5))


def test_image_query_returns_captions_aggregated_by_item(engine):
    hits = engine.search(Query("image", png_bytes(7), k=5))
    assert hits[0].item_id == "img_000007" and hits[0].caption.startswith("fake caption 7")
    assert len({h.item_id for h in hits}) == 5
    modal = engine.search(Query("image", png_bytes(7), k=3, mode="modal"))
    assert modal[0].item_id == "img_000007" and modal[0].scores_by_modality[
        "image"
    ] == pytest.approx(1.0, abs=1e-4)


def test_audio_query_modes(engine):
    fused = engine.search(Query("audio", b"12", k=5, suffix=".wav"))
    assert fused[0].item_id == "aud_000012"
    modal = engine.search(Query("audio", b"12", k=5, mode="modal", suffix=".wav"))
    assert modal[0].item_id == "aud_000012" and modal[0].caption.startswith("fake sound 12")


def test_classify_audio(engine):
    top = engine.classify_audio(b"3", classes=None, suffix=".wav")
    assert (
        len(top) == 5
        and abs(sum(p for _, p in engine.classify_audio(b"3", suffix=".wav", topk=10)) - 1) < 1e-5
    )
    assert top[0][1] >= top[-1][1]


@pytest.mark.parametrize(
    "q",
    [
        Query("text", "  "),
        Query("text", "x" * 501),
        Query("text", "a", k=0),
        Query("text", "a", k=51),
        Query("text", "a", mode="zzz"),
        Query("text", "a", filters={"bad": 1}),
        Query("text", "a", filters={"modality": "video"}),
        Query("video", "a"),
    ],
)
def test_invalid_queries(engine, q):
    with pytest.raises(QueryError):
        engine.search(q)


def test_errors_and_capabilities(cfg, engine):
    with pytest.raises(MediaError):
        engine.search(Query("audio", b"pas un nombre", suffix=".wav"))
    with pytest.raises(MediaError):
        engine.search(Query("image", b"pas une image"))
    with pytest.raises(NotFound):
        engine.get_item("nope")
    only_text = SearchEngine.from_artifacts(cfg, encoders={"text": lambda: None})
    assert only_text.capabilities() == {"text": True, "image": False, "audio": False}
    with pytest.raises(CapabilityError):
        only_text.search(Query("audio", b"1"))
    assert engine.get_item("img_000001")["captions"][0]["text"] == "fake caption 1 0"
    p = engine.media_path("img_000001")
    assert p.name == "fake_000001.jpg" and "data" in p.parts
    engine.items["img_000001"]["path"] = "../../etc/passwd"
    with pytest.raises(NotFound):
        engine.media_path("img_000001")
    assert np.isfinite(engine.stats()["n_items"])
