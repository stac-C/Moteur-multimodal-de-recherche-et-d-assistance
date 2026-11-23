import numpy as np
import pytest

from mmsearch.eval.metrics import (
    first_relevant_rank,
    mrr,
    mrr_from_ranks,
    precision_at_k,
    ranks_from_scores,
    recall_at_k,
    recall_from_ranks,
)


def test_section_1_5_recall_and_mrr():
    ranks = np.array([1, 3, 7, 12])
    assert recall_from_ranks(ranks, 1) == 0.25
    assert recall_from_ranks(ranks, 5) == 0.5
    assert recall_from_ranks(ranks, 10) == 0.75
    assert mrr_from_ranks(ranks) == pytest.approx(0.3899, abs=1e-4)


def test_precision_example():
    ranked = [[f"c{i}" for i in range(10)]]
    relevant = [{"c0", "c3", "c5", "c9", "zz"}]  # |R| = 5, 4 dans le top-10
    assert precision_at_k(ranked, relevant, 10) == pytest.approx(0.4)
    assert recall_at_k(ranked, relevant, 10, mode="fraction") == pytest.approx(0.8)
    assert recall_at_k(ranked, relevant, 10, mode="hit") == 1.0
    assert mrr(ranked, relevant) == 1.0


def test_ranks_from_scores_and_first_relevant():
    s = np.array([[0.1, 0.9, 0.5], [0.3, 0.2, 0.8]])
    assert ranks_from_scores(s, np.array([2, 0])).tolist() == [2, 2]
    rel = np.array([[True, False, True], [False, True, False]])
    assert first_relevant_rank(s, rel).tolist() == [2, 3]
