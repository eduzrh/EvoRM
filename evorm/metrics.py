#!/usr/bin/env python3
"""
Optional metric helpers.

EvoRM does **not** define a new evaluation protocol. Your framework already
reports Hits@1 / MRR / F1 / accuracy on its own test split -- keep using those
numbers. This module only exists so that (a) the bundled experiment scripts run
standalone and (b) the EvoRM-specific efficiency statistics (routing rate, token
consumption, latency) have a canonical definition.

If your framework has its own evaluator, ignore everything here except
``efficiency_summary``.
"""

from typing import Dict, Iterable, List, Sequence


# ---------------------------------------------------------------- ranking (EA / EL)
def hits_at_k(ranks: Sequence[int], k: int) -> float:
    """Fraction of queries whose gold entity is ranked within the top ``k``.

    ``ranks`` holds 1-based ranks; use ``len(candidates) + 1`` for a miss.
    """
    if not ranks:
        return 0.0
    return sum(1 for r in ranks if 1 <= r <= k) / len(ranks)


def mrr(ranks: Sequence[int]) -> float:
    """Mean reciprocal rank of the gold entity."""
    if not ranks:
        return 0.0
    return sum(1.0 / r for r in ranks if r > 0) / len(ranks)


def rank_of(gold_id: str, ranked_ids: Sequence[str]) -> int:
    """1-based rank of ``gold_id`` in ``ranked_ids``; len + 1 when absent."""
    for i, cid in enumerate(ranked_ids, 1):
        if str(cid) == str(gold_id):
            return i
    return len(ranked_ids) + 1


# ---------------------------------------------------------------- binary (ER / SM)
def precision_recall_f1(gold: Sequence[int], pred: Sequence[int]) -> Dict[str, float]:
    """Binary F1 with the positive class as 'match', as used for ER."""
    tp = sum(1 for g, p in zip(gold, pred) if g == 1 and p == 1)
    fp = sum(1 for g, p in zip(gold, pred) if g == 0 and p == 1)
    fn = sum(1 for g, p in zip(gold, pred) if g == 1 and p == 0)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn}


def accuracy(gold: Sequence[int], pred: Sequence[int]) -> float:
    """Plain accuracy; equals F1 in the many-to-one SM setup."""
    if not gold:
        return 0.0
    return sum(1 for g, p in zip(gold, pred) if g == p) / len(gold)


# ---------------------------------------------------------------- entity linking
def normalized_accuracy(gold_in_candidates: Sequence[bool],
                        correct: Sequence[bool]) -> Dict[str, float]:
    """EL accuracy normalised over mentions whose gold is in the candidate set.

    Follows the ZESHEL protocol: the denominator only counts mentions whose gold
    entity is present in the provided top-64 BM25 candidates.
    """
    n_applicable = sum(1 for g in gold_in_candidates if g)
    n_correct = sum(1 for g, c in zip(gold_in_candidates, correct) if g and c)
    return {
        "accuracy": n_correct / n_applicable if n_applicable else 0.0,
        "coverage": n_applicable / len(gold_in_candidates) if gold_in_candidates else 0.0,
        "n_applicable": n_applicable,
    }


def macro_average(scores: Dict[str, float]) -> float:
    """Unweighted mean across domains (paper: macro accuracy over 4 ZESHEL domains)."""
    return sum(scores.values()) / len(scores) if scores else 0.0


# ---------------------------------------------------------------- efficiency
def efficiency_summary(records: Iterable[Dict], baseline_tokens: float = 0.0) -> Dict:
    """EvoRM-specific efficiency statistics.

    ``records`` items must expose ``tokens``, ``latency_s`` and ``stage``.
    ``baseline_tokens`` is the average per-query token cost of the un-augmented
    framework, used for the relative saving reported in RQ4.
    """
    records = list(records)
    n = len(records)
    if n == 0:
        return {"queries": 0, "avg_tokens": 0.0, "avg_latency_s": 0.0,
                "routing_rate": 0.0, "token_saving": 0.0}

    tokens = sum(r.get("tokens", 0) for r in records)
    latency = sum(r.get("latency_s", 0.0) for r in records)
    stage1 = sum(1 for r in records if r.get("stage") == "stage1")
    avg_tokens = tokens / n

    saving = 1.0 - avg_tokens / baseline_tokens if baseline_tokens > 0 else 0.0
    return {
        "queries": n,
        "avg_tokens": avg_tokens,
        "avg_latency_s": latency / n,
        "routing_rate": stage1 / n,
        "token_saving": saving,
    }


def format_table(rows: List[Dict], columns: Sequence[str]) -> str:
    """Plain-text table used by the experiment scripts."""
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) if rows else len(c)
              for c in columns}
    header = "  ".join(c.ljust(widths[c]) for c in columns)
    line = "  ".join("-" * widths[c] for c in columns)
    body = ["  ".join(str(r.get(c, "")).ljust(widths[c]) for c in columns) for r in rows]
    return "\n".join([header, line] + body)