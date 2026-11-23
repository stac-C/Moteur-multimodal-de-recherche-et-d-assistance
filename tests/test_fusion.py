import numpy as np
import pytest

from mmsearch.fusion.late import (
    caption_context,
    evaluate_text_to_item,
    grid_search_weights,
    late_fusion,
    text_to_item_scores_loo,
)


def unit(x):
    return (x / np.linalg.norm(x, axis=-1, keepdims=True)).astype(np.float32)


def make(n=60, K=5, d=64, noise_img=0.5, noise_cap=0.9, seed=0):
    rng = np.random.default_rng(seed)
    base = unit(rng.normal(size=(n, d)))
    img = unit(base + noise_img * rng.normal(size=base.shape) / np.sqrt(d) * 4)
    cap = unit(base[:, None, :] + noise_cap * rng.normal(size=(n, K, d)) / np.sqrt(d) * 4)
    return img, cap


def test_late_fusion_linear_and_checks():
    a, b = np.ones((2, 3), np.float32), 2 * np.ones((2, 3), np.float32)
    out = late_fusion({"x": a, "y": b}, {"x": 0.25, "y": 0.75})
    assert out.dtype == np.float32 and np.allclose(out, 1.75)
    with pytest.raises(ValueError):
        late_fusion({"x": a, "y": b}, {"x": 0.5, "y": 0.6})
    with pytest.raises(ValueError):
        late_fusion({"x": a}, {"x": 0.5, "y": 0.5})


def test_caption_context_leave_one_out():
    _, cap = make(n=4)
    full, loo = caption_context(cap)
    manual = cap[1, [0, 1, 3, 4]].mean(axis=0)
    manual /= np.linalg.norm(manual)
    assert np.allclose(loo[1, 2], manual, atol=1e-6)
    assert np.allclose(np.linalg.norm(full, axis=1), 1.0, atol=1e-5)


def test_loo_scores_replace_only_owner_column():
    img, cap = make(n=6)
    n, K, _ = cap.shape
    full, loo = caption_context(cap)
    owner = np.repeat(np.arange(n), K)
    z_full, z_loo = 0.7 * img + 0.3 * full, 0.7 * img[owner] + 0.3 * loo.reshape(n * K, -1)
    q = cap.reshape(n * K, -1)
    s = text_to_item_scores_loo(q, z_full, z_loo, owner)
    plain = q @ z_full.T
    mask = np.ones_like(s, dtype=bool)
    mask[np.arange(n * K), owner] = False
    assert np.allclose(s[mask], plain[mask])
    assert np.allclose(s[np.arange(n * K), owner], np.einsum("md,md->m", q, z_loo), atol=1e-6)


def test_fusion_beats_chance_and_grid_search():
    img, cap = make(n=80)
    m = evaluate_text_to_item(lambda a, b: 0.7 * a + 0.3 * b, img, cap)
    assert m["R@10"] > 10 / 80 * 3  # bien au-dessus du hasard
    w, rows = grid_search_weights(img, cap, step=0.25)
    assert 0.0 <= w <= 1.0 and len(rows) == 5
    assert max(r["rsum"] for r in rows) >= rows[0]["rsum"]
