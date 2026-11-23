import json

import numpy as np
import pytest

from mmsearch.eval import bias_analysis, make_figures, run_eval
from mmsearch.eval.run_eval import latex_table, proxy_precision
from mmsearch.viz.projections import build_projection


@pytest.fixture(scope="module")
def results(cfg):
    return run_eval.run(cfg, "test", w=0.7)


def test_run_eval_outputs(cfg, results):
    t2i = results["texte->image (Flickr8k)"]
    assert "mono-texte (SBERT, leave-one-out)" in t2i and "mono-image (CLIP)" in t2i
    fused = t2i["fusion tardive (w_img = 0.7)"]
    assert fused["R@10"] >= t2i["mono-image (CLIP)"]["R@10"] - 1e-9 and fused["R@1"] > 1 / 12
    assert set(results["ablation dimension d (fusion tardive)"]) == {"128", "256", "512"}
    a = results["audio<->texte (Clotho)"]["AST + MLP (E5)"]
    assert 0 < a["item->texte"]["P@5"] <= 1 and a["texte->item"]["MedR"] >= 1
    f = cfg.artifacts / "figures"
    for name in ("results.json", "results.csv", "results_tables.tex", "case_studies.json"):
        assert (f / name).exists()
    assert r"\toprule" in (f / "results_tables.tex").read_text(encoding="utf-8")
    cases = json.loads((f / "case_studies.json").read_text(encoding="utf-8"))
    assert cases and {c["outcome"] for c in cases} <= {"succès", "échec"}


def test_flat_and_latex(results):
    rows = run_eval.flatten(results)
    assert ("texte->image (Flickr8k)", "mono-image (CLIP)", "R@1") in [r[:3] for r in rows]
    tex = latex_table(
        {"a_b": {"R@1": 0.5, "R@5": 0.6, "R@10": 0.7, "MRR": 0.55, "MedR": 2.0}},
        ["R@1", "MRR", "MedR"],
        "x & y",
        "tab:t",
    )
    assert "a\\_b & 50.0 & 0.550 & 2" in tex and "x \\& y" in tex


def test_proxy_precision_pure():
    texts = [
        "a dog runs",
        "a cat sleeps",
        "a dog barks",
        "man",
        "dogs play",
        "dog",
        "dog park",
        "x",
    ]
    S = np.array(
        [[0.9, 0.1, 0.8, 0.0, 0.7, 0.6, 0.5, 0.0]]
    )  # un clip « dog » proche des images « dog »
    r = proxy_precision(
        S, ["dog"], texts, {"dog": ["dog", "dogs"], "cat": ["cat"]}, k=3, min_relevant=3
    )
    assert set(r["classes"]) == {"dog"}  # « cat » : 1 image pertinente < 3
    c = r["classes"]["dog"]
    assert (
        c["P@3"] == 1.0
        and c["base_rate"] == pytest.approx(5 / 8)
        and c["lift"] == pytest.approx(1.6)
    )


def test_bias_pure_functions():
    assert bias_analysis.groups_of("A man and a dog on the beach") == [
        "hommes",
        "animaux",
        "nature",
    ]
    ranks = {"s": np.array([1, 2, 20, 1, 1, 30])}
    r = bias_analysis.group_retrieval(
        ["a man", "a man", "a man", "a woman", "a woman", "a woman"], ranks, min_n=3
    )
    assert r["groups"]["hommes"]["s R@10"] == pytest.approx(2 / 3) and r["groups"]["femmes"][
        "s R@1"
    ] == pytest.approx(2 / 3)
    assert r["disparity_R@10"]["s"]["gap"] == pytest.approx(0.0)


def test_make_figures(cfg, results):
    build_projection(cfg, "pca", per_group=40)
    made = {p.name for p in make_figures.main(cfg)}
    assert {"fig_mono_vs_multi.png", "fig_dim_ablation.png", "fig_proj_pca.png"} <= made
    assert all((cfg.artifacts / "figures" / n).stat().st_size > 1000 for n in made)
