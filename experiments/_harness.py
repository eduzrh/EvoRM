#!/usr/bin/env python3
"""Shared experiment harness: dataset -> port -> records -> metrics."""

import os
import statistics
import sys
import time
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evorm.dataset import Candidate, PairTask, RankingTask, is_ranking, load_tasks  # noqa: E402
from evorm.metrics import (accuracy, efficiency_summary, hits_at_k, mrr,  # noqa: E402
                           precision_recall_f1, rank_of)
from evorm.port import EvoRMPort  # noqa: E402
from evorm_core.config import EvoRMConfig  # noqa: E402


def make_client(model: Optional[str] = None):
    """Create an OpenAI-compatible client from environment variables.

    OPENAI_API_KEY  -- required
    OPENAI_BASE_URL -- optional, defaults to https://api.openai.com/v1
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise SystemExit("pip install openai  (or pass --no-llm for symbolic-only runs)")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set")
    return OpenAI(
        api_key=api_key,
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )


def build_port(cfg: EvoRMConfig, use_llm: bool = True,
               base_matcher=None) -> EvoRMPort:
    client = make_client(cfg.model) if use_llm else None
    return EvoRMPort(config=cfg, client=client, mlight_client=client,
                     base_matcher=base_matcher)


# ---------------------------------------------------------------- binary tasks
def run_pairs(port: EvoRMPort, tasks: List[PairTask]) -> List[Dict]:
    records = []
    for task in tasks:
        out = port.decide(task.es_id, task.et_id, task.es_context, task.et_context)
        records.append({
            "label": task.label,
            "pred": out.label,
            "stage": out.stage,
            "tokens": out.tokens,
            "latency_s": out.latency_s,
        })
    return records


def summarize_pairs(records: List[Dict], baseline_tokens: float = 0.0) -> Dict:
    gold = [r["label"] for r in records]
    pred = [r["pred"] for r in records]
    prf = precision_recall_f1(gold, pred)
    out = {
        "n": len(records),
        "accuracy": accuracy(gold, pred),
        "f1": prf["f1"],
        "precision": prf["precision"],
        "recall": prf["recall"],
    }
    out.update(efficiency_summary(records, baseline_tokens))
    return out


# ---------------------------------------------------------------- ranking tasks
def run_rankings(port: EvoRMPort, tasks: List[RankingTask],
                 cold_start: int = 0) -> List[Dict]:
    """Cold-start on the first ``cold_start`` tasks, then evaluate the rest."""
    records = []

    for idx, task in enumerate(tasks):
        if idx < cold_start:
            # Cold start: let the LLM resolve the top-1 candidate so that rules
            # accumulate before the symbolic reranker is used for scoring.
            if not task.candidates:
                continue
            top = max(task.candidates, key=lambda c: c.base_score)
            port.decide(task.query_id, top.entity_id,
                        task.query_context, top.context,
                        prompt=_cold_start_prompt(task, top))
            continue

        started = time.time()
        scores = {c.entity_id: c.base_score for c in task.candidates}
        ranked = port.rerank(task.query_id, task.query_context,
                             [(c.entity_id, c.context) for c in task.candidates],
                             base_scores=scores)
        latency = time.time() - started

        ranked_ids = [r.entity_id for r in ranked]
        rank = rank_of(task.gold_id, ranked_ids) if task.gold_id else len(ranked_ids) + 1
        records.append({
            "rank": rank,
            "stage": "stage1",
            "tokens": 0,
            "latency_s": latency,
            "num_candidates": len(ranked_ids),
        })
    return records


def _cold_start_prompt(task: RankingTask, cand: Candidate) -> str:
    return (
        "Decide whether the two entities refer to the same real-world object.\n"
        f"Entity A: {task.query_context}\nEntity B: {cand.context}\n"
        "Answer MATCH or NON-MATCH."
    )


def summarize_rankings(records: List[Dict]) -> Dict:
    ranks = [r["rank"] for r in records]
    n_cand = [r["num_candidates"] for r in records]
    return {
        "n": len(records),
        "hits@1": hits_at_k(ranks, 1),
        "hits@10": hits_at_k(ranks, 10),
        "mrr": mrr(ranks),
        "avg_candidates": statistics.mean(n_cand) if n_cand else 0,
        "avg_latency_s": statistics.mean([r["latency_s"] for r in records]) if records else 0.0,
    }


# ---------------------------------------------------------------- entry helper
def load_and_run(path: str, port: EvoRMPort, limit: Optional[int] = None,
                 cold_start: int = 0) -> Dict:
    tasks = load_tasks(path, limit=limit)
    if not tasks:
        return {"n": 0}
    if is_ranking(tasks):
        records = run_rankings(port, tasks, cold_start=cold_start)
        return summarize_rankings(records)
    records = run_pairs(port, tasks)
    return summarize_pairs(records)