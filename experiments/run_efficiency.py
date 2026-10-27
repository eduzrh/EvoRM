#!/usr/bin/env python3
"""
RQ4 -- accuracy / efficiency trade-off (paper Table VII).

Reports the token and latency cost of EvoRM relative to sending every pair to
the LLM, i.e. the ``w/o Stage-1`` configuration used as the cost baseline.

    python experiments/run_efficiency.py --data data/dbp15k_en_fr.jsonl --cold-start 500
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _harness import build_port, load_and_run  # noqa: E402
from evorm.metrics import format_table  # noqa: E402
from evorm_core.config import EvoRMConfig, ablation_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default=EvoRMConfig.model)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--cold-start", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    results = {}
    for name in ("Full System (EvoRM)", "w/o Stage-1 (cost baseline)"):
        cfg = (EvoRMConfig(model=args.model) if name.startswith("Full")
               else ablation_config("w/o Stage-1", EvoRMConfig(model=args.model)))
        port = build_port(cfg)
        results[name] = load_and_run(args.data, port, limit=args.limit,
                                     cold_start=args.cold_start)

    base = results["w/o Stage-1 (cost baseline)"]
    full = results["Full System (EvoRM)"]

    def pct(new, old):
        return (1.0 - new / old) if old else 0.0

    rows = []
    for name, r in results.items():
        rows.append({
            "configuration": name,
            "metric": round(float(r.get("hits@1", r.get("f1", r.get("accuracy", 0.0)))), 4),
            "avg_tokens": round(float(r.get("avg_tokens", 0.0)), 1),
            "avg_latency_s": round(float(r.get("avg_latency_s", 0.0)), 2),
            "routing_rate": round(float(r.get("routing_rate", 0.0)), 4),
        })

    print(format_table(rows, ["configuration", "metric", "avg_tokens",
                              "avg_latency_s", "routing_rate"]))
    print(f"\nToken reduction   : {pct(full.get('avg_tokens', 0), base.get('avg_tokens', 0)):.1%}")
    print(f"Latency reduction : {pct(full.get('avg_latency_s', 0), base.get('avg_latency_s', 0)):.1%}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()