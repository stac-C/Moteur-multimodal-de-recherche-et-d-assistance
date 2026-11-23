import numpy as np
import pytest

from mmsearch.index.store import VectorStore


def unit(x):
    return (x / np.linalg.norm(x, axis=-1, keepdims=True)).astype(np.float32)


def test_search_matches_bruteforce_and_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    v, q = unit(rng.normal(size=(300, 512))), unit(rng.normal(size=(5, 512)))
    ids = [f"x{i}" for i in range(300)]
    st = VectorStore(512)
    st.build(v, ids)
    scores, res = st.search(q, 7)
    brute = np.argsort(-(q @ v.T), axis=1)[:, :7]
    assert [[ids[j] for j in row] for row in brute] == res
    assert np.allclose(scores, np.sort(q @ v.T, axis=1)[:, ::-1][:, :7], atol=1e-5)
    st.save(str(tmp_path / "t.faiss"))
    st2 = VectorStore.load(str(tmp_path / "t.faiss"))
    assert st2.search(q, 7)[1] == res


def test_filter_and_errors():
    rng = np.random.default_rng(1)
    v = unit(rng.normal(size=(100, 512)))
    ids = [f"x{i}" for i in range(100)]
    st = VectorStore(512)
    st.build(v, ids)
    allowed = {f"x{i}" for i in range(0, 100, 2)}
    scores, res = st.search(v[:3], 5, id_filter=allowed)
    assert all(i in allowed for row in res for i in row)
    assert res[0][0] == "x0" and scores[0, 0] == pytest.approx(1.0, abs=1e-5)
    assert st.search(v[:1], 5, id_filter={"nope"})[1] == [[]]
    assert len(st.search(v[:1], 500)[1][0]) == 100  # k borné au nombre d'items
    with pytest.raises(ValueError):
        st.build(v, ids[:-1])
    with pytest.raises(ValueError):
        st.search(np.zeros((1, 8), np.float32), 3)
