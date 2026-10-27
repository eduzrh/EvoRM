"""Optional metric helpers."""

import pytest

from evorm.metrics import (accuracy, efficiency_summary, hits_at_k, macro_average,
                           mrr, normalized_accuracy, precision_recall_f1, rank_of)


def test_hits_and_mrr():
    ranks = [1, 3, 12, 50]
    assert hits_at_k(ranks, 1) == pytest.approx(0.25)
    assert hits_at_k(ranks, 10) == pytest.approx(0.50)
    assert mrr(ranks) == pytest.approx((1 + 1 / 3 + 1 / 12 + 1 / 50) / 4)


def test_rank_of():
    assert rank_of("c", ["a", "b", "c"]) == 3
    assert rank_of("x", ["a", "b"]) == 3


def test_f1_and_accuracy():
    gold = [1, 1, 0, 0]
    pred = [1, 0, 0, 1]
    m = precision_recall_f1(gold, pred)
    assert m["tp"] == 1 and m["fp"] == 1 and m["fn"] == 1
    assert m["f1"] == pytest.approx(0.5)
    assert accuracy(gold, pred) == pytest.approx(0.5)


def test_normalized_accuracy_ignores_missing_gold():
    m = normalized_accuracy([True, False, True, True], [True, True, False, True])
    assert m["n_applicable"] == 3
    assert m["accuracy"] == pytest.approx(2 / 3)


def test_macro_average_over_domains():
    assert macro_average({"a": 0.9, "b": 0.7}) == pytest.approx(0.8)


def test_efficiency_summary():
    records = [{"tokens": 100, "latency_s": 1.0, "stage": "stage1"},
               {"tokens": 300, "latency_s": 3.0, "stage": "stage2"}]
    out = efficiency_summary(records, baseline_tokens=400)
    assert out["avg_tokens"] == pytest.approx(200)
    assert out["routing_rate"] == pytest.approx(0.5)
    assert out["token_saving"] == pytest.approx(0.5)
