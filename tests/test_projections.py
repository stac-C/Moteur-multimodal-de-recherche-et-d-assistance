import numpy as np
import pytest

from mmsearch.viz.projections import build_joint, build_projection, load_or_compute, pca_2d, project


def test_pca_2d_recovers_principal_axes():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5)) * np.array([5, 2, 0.1, 0.1, 0.1])
    xy, ev = pca_2d(X)
    assert xy.shape == (200, 2) and ev[0] > ev[1] > 0 and sum(ev) > 0.99


def test_build_and_load(cfg):
    X, meta = build_joint(cfg, per_group=50, seed=1)
    assert X.shape[0] == len(meta) == 50 + 50 + 40 + 30 and X.shape[1] == 512
    assert {m["modality"] for m in meta} == {"image", "caption", "audio"}
    path = build_projection(cfg, "pca", per_group=50)
    assert path.exists() and load_or_compute(cfg, "pca")["n_points"] == 170
    with pytest.raises(FileNotFoundError):
        load_or_compute(cfg, "tsne")  # t-SNE doit être précalculé hors-ligne
    xy, _ = project(X[:80], "tsne", seed=0)
    assert xy.shape == (80, 2) and np.isfinite(xy).all()
