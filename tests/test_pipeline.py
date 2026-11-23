"""Tests de contrat (A6) : artefacts factices -> index -> SQLite, formats conformes à la section 2.7."""

import sqlite3

import numpy as np
import pytest

from mmsearch.config import load_config
from mmsearch.data.loaders import captions_tensor, load_captions, load_manifest
from mmsearch.embeddings_io import check_embeddings, load_embeddings
from mmsearch.fake import make_fake_artifacts
from mmsearch.index.build_index import build_all
from mmsearch.index.store import VectorStore
from mmsearch.tasks.zero_shot_cls import (
    classify_one,
    confusion_matrix,
    class_prompts,
    topk_accuracy,
)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = make_fake_artifacts(tmp_path_factory.mktemp("fake"), n_img=60, n_aud=40, n_esc=30)
    cfg = load_config(root / "config.yaml")
    return cfg, build_all(cfg)


def test_index_sizes(built):
    cfg, sizes = built
    assert sizes == {"idx_img": 60, "idx_aud": 40, "idx_txt": 500, "idx_fused": 100}


def test_manifest_schema(built):
    cfg, _ = built
    m = load_manifest(cfg)
    assert list(m.columns) == [
        "item_id",
        "modality",
        "dataset",
        "split",
        "path",
        "label",
        "duration_s",
    ]
    assert m["item_id"].is_unique and set(m["split"]) <= {"train", "val", "test"}
    c = load_captions(cfg)
    assert list(c.columns) == ["caption_id", "item_id", "idx", "text"] and len(c) == 500


def test_embedding_contract(built):
    cfg, _ = built
    for f in [
        "clip_img_flickr8k",
        "clip_txt_flickr8k",
        "audio_clotho",
        "clip_txt_clotho",
        "audio_esc50",
    ]:
        arr, ids = load_embeddings(cfg.emb_dir / f"{f}.npy")
        assert arr.dtype == np.float32 and arr.shape == (len(ids), 512)
    fused, _ = load_embeddings(cfg.emb_dir / "fused_items.npy", max_norm=True)
    assert np.linalg.norm(fused, axis=1).max() <= 1.0 + 1e-4
    bad = np.ones((2, 512), np.float32)
    with pytest.raises(ValueError):
        check_embeddings(bad, ["a", "b"])  # normes != 1


def test_text_to_image_retrieval_and_sqlite(built):
    cfg, _ = built
    store = VectorStore.load(str(cfg.index_dir / "idx_img.faiss"))
    cap_emb, cap_ids = load_embeddings(cfg.emb_dir / "clip_txt_flickr8k.npy")
    q = cap_emb[::5][:20]  # 1re légende de 20 images
    _, res = store.search(q, 10)
    hits = sum(f"img_{i:06d}" in res[i] for i in range(20))
    assert hits >= 15  # corrélation forte dans les données factices
    con = sqlite3.connect(cfg.index_dir / "collection.db")
    assert con.execute("SELECT COUNT(*) FROM items").fetchone()[0] == 130
    assert con.execute("SELECT COUNT(*) FROM captions").fetchone()[0] == 500
    con.close()
    t = captions_tensor(["img_000000"], load_captions(cfg), cap_emb, cap_ids)
    assert t.shape == (1, 5, 512)


def test_zero_shot_helpers():
    rng = np.random.default_rng(0)
    cls = rng.normal(size=(5, 16))
    cls /= np.linalg.norm(cls, axis=1, keepdims=True)
    y = np.array([0, 1, 2, 3, 4, 0])
    audio = cls[y] + 0.05 * rng.normal(size=(6, 16))
    audio /= np.linalg.norm(audio, axis=1, keepdims=True)
    s = audio @ cls.T
    assert topk_accuracy(s, y, 1) == 1.0 and topk_accuracy(s, y, 5) == 1.0
    assert np.trace(confusion_matrix(y, s.argmax(1), 5)) == 6
    top = classify_one(audio[1], cls, ["a", "b", "c", "d", "e"], topk=2)
    assert (
        top[0][0] == "b"
        and abs(sum(p for _, p in classify_one(audio[1], cls, list("abcde"), topk=5)) - 1) < 1e-6
    )
    assert class_prompts(["pouring_water"]) == ["the sound of pouring water"]
