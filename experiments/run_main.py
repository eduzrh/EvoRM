#!/usr/bin/env python3
"""
RQ1 / RQ2 -- main results and cross-backbone generality.

Runs EvoRM over a JSONL task file and reports the task's standard metric plus
EvoRM's efficiency statistics. Swap ``--model`` to reproduce the RQ2 comparison
across backbone LLMs.

    python experiments/run_main.py --data data/abt.jsonl --task er --model meta-llama/Llama-3.1-8B-Instruct
    python experiments/run_main.py --data data/dbp15k_en_fr.jsonl --task ea --cold-start 500
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _harness import build_port, load_and_run  # noqa: E402
from evorm_core.config import EvoRMConfig  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="JSONL task file")
    ap.add_argument("--task", default="er", choices=["ea", "er", "el", "sm"])
    ap.add_argument("--model", default=EvoRMConfig.model)
    ap.add_argument("--mlight-model", default=EvoRMConfig.mlight_model)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--cold-start", type=int, default=0,
                    help="trajectories used to warm up the rule bank (ranking tasks)")
    ap.add_argument("--no-llm", action="store_true",
                    help="symbolic-only run (no LLM calls, rules never learned)")
    ap.add_argument("--out", default=None, help="optional JSON output path")
    args = ap.parse_args()

    cfg = EvoRMConfig(model=args.model, mlight_model=args.mlight_model)
    port = build_port(cfg, use_llm=not args.no_llm)

    summary = load_and_run(args.data, port, limit=args.limit,
                           cold_start=args.cold_start)
    summary["task"] = args.task
    summary["model"] = args.model
    summary["stats"] = port.stats()

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()