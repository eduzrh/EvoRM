#!/usr/bin/env python3
"""
RQ5 -- rule-bank scaling with matching experience (paper Table VIII).

For each t in {100, 300, 500, 1000}, EvoRM ingests the first t unlabeled
trajectories, freezes the hypergraph, and is then evaluated on the test split.

    python experiments/run_scaling.py \
        --train data/dbp15k_en_fr_train.jsonl \
        --test  data/dbp15k_en_fr_test.jsonl \
        --trajectories 100 300 500 1000
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _harness import build_port, load_and_run  # noqa: E402
from evorm.dataset import is_ranking, load_tasks  # noqa: E402
from evorm.metrics import format_table  # noqa: E402
from evorm_core.config import EvoRMConfig  # noqa: E402


def ingest(port, train_path: str, t: int):
    """Warm the rule bank on the first ``t`` unlabeled trajectories."""
    tasks = load_tasks(train_path, limit=t)
    if not tasks:
        return
    if is_ranking(tasks):
        for task in tasks:
            if not task.candidates:
                continue
            top = max(task.candidates, key=lambda c: c.base_score)
            port.decide(task.query_id, top.entity_id,
                        task.query_context, top.context)
    else:
        for task in tasks:
            port.decide(task.es_id, task.et_id, task.es_context, task.et_context)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="unlabeled trajectories (JSONL)")
    ap.add_argument("--test", required=True, help="evaluation split (JSONL)")
    ap.add_argument("--trajectories", type=int, nargs="+",
                    default=[100, 300, 500, 1000])
    ap.add_argument("--model", default=EvoRMConfig.model)
    ap.add_argument("--limit", type=int, default=None, help="cap the test split")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = []
    for t in args.trajectories:
        port = build_port(EvoRMConfig(model=args.model))
        ingest(port, args.train, t)
        summary = load_and_run(args.test, port, limit=args.limit)
        rows.append({
            "trajectories": t,
            "accuracy": round(float(summary.get("hits@1",
                                                summary.get("f1",
                                                            summary.get("accuracy", 0.0)))), 4),
            "avg_tokens": round(float(summary.get("avg_tokens", 0.0)), 1),
            "num_rules": port.plugin.hypergraph.stats()["num_rules"],
            "num_hyperedges": port.plugin.hypergraph.stats()["num_hyperedges"],
            "routing_rate": round(float(summary.get("routing_rate", 0.0)), 4),
        })
        print(json.dumps(rows[-1], ensure_ascii=False))

    print()
    print(format_table(rows, ["trajectories", "accuracy", "avg_tokens",
                              "num_rules", "num_hyperedges", "routing_rate"]))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()