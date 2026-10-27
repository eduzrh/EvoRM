#!/usr/bin/env python3
"""
RQ3 -- component ablation (paper Table VI).

Configurations
    full          EvoRM with all four components
    w/o U         weighted hypergraph replaced by a flat relational layout
    w/o G         rule maintenance disabled (no evolution)
    w/o Stage-1   symbolic filtering disabled; every pair goes to the LLM

    python experiments/run_ablation.py \
        --ea data/dbp15k_en_fr.jsonl \
        --er data/abt.jsonl \
        --el data/zeshel_yugioh.jsonl
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _harness import build_port, load_and_run  # noqa: E402
from evorm.metrics import format_table  # noqa: E402
from evorm_core.config import ABLATION_CONFIGS, EvoRMConfig, ablation_config  # noqa: E402

METRIC_BY_TASK = {"ea": "hits@1", "er": "f1", "el": "hits@1", "sm": "accuracy"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ea", help="EA JSONL (ranking)")
    ap.add_argument("--er", help="ER JSONL (binary)")
    ap.add_argument("--el", help="EL JSONL (ranking)")
    ap.add_argument("--model", default=EvoRMConfig.model)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--cold-start", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    datasets = [(t, p) for t, p in (("ea", args.ea), ("er", args.er), ("el", args.el)) if p]
    if not datasets:
        raise SystemExit("provide at least one of --ea / --er / --el")

    results = {}
    for name in ABLATION_CONFIGS:
        cfg = ablation_config(name, EvoRMConfig(model=args.model))
        port = build_port(cfg)
        row = {}
        for task, path in datasets:
            summary = load_and_run(path, port, limit=args.limit,
                                   cold_start=args.cold_start)
            row[task] = summary.get(METRIC_BY_TASK[task], 0.0)
            row[f"{task}_routing"] = summary.get("routing_rate", 0.0)
            row[f"{task}_tokens"] = summary.get("avg_tokens", 0.0)
        results[name] = row
        print(f"[{name}] " + json.dumps(row, ensure_ascii=False))

    columns = ["configuration"] + [t for t, _ in datasets]
    rows = [{"configuration": k, **{t: round(v.get(t, 0.0), 4) for t, _ in datasets}}
            for k, v in results.items()]
    print("\n" + format_table(rows, columns))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()