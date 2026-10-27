#!/usr/bin/env python3
"""End-to-end demo that runs offline: no API key, no downloads, ~2 seconds.

It exercises the whole loop on a synthetic ER stream:

    Stage-1 symbolic routing -> Stage-2 host matcher -> rule encoding
    -> hypergraph storage -> confidence update (Eq. 8) -> maintenance

The host matcher here is a deterministic attribute comparator standing in for
your model.  Swap it for a call to GPT / Llama / your own matcher and the rest
of the script is unchanged -- that is the point of the port.

Run:

    python examples/quickstart.py
"""

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evorm.metrics import efficiency_summary, precision_recall_f1  # noqa: E402
from evorm.port import EvoRMPort  # noqa: E402
from evorm_core.config import EvoRMConfig  # noqa: E402

CFG = EvoRMConfig(enable_mlp_gate=False)   # the gate needs torch; not needed here

COUNTRIES = ["US", "DE", "FR", "JP", "BR"]
SUFFIX = ["Inc", "Ltd", "GmbH", "SA", "KK"]
STEMS = ["Acme", "Globex", "Initech", "Umbrella", "Hooli", "Vehement",
         "Massive", "Vandelay", "Soylent", "Initrode", "Bluth", "Prestige"]


def build_stream(n=120, seed=7):
    """Synthetic company records with a 50/50 match rate."""
    rng = random.Random(seed)
    records, pairs = [], []
    for i in range(n):
        stem = rng.choice(STEMS)
        country = rng.choice(COUNTRIES)
        suffix = rng.choice(SUFFIX)
        rec = {"name": f"{stem} {suffix}", "country": country,
               "founded": str(rng.randint(1950, 2015)),
               "website": f"www.{stem.lower()}.com"}
        records.append(rec)

        if i % 2 == 0:                        # duplicate with one noisy field
            twin = dict(rec)
            noise = rng.random()
            if noise < 0.35:                  # domain migrated
                twin["website"] = f"www.{stem.lower()}-group.com"
            elif noise < 0.70:                # incorporation year off by one
                twin["founded"] = str(int(rec["founded"]) + rng.choice([-1, 1]))
            pairs.append({"id": f"p{i}", "a": rec, "b": twin, "gold": 1})
        elif rng.random() < 0.25:             # hard negative: shares country + year
            other = {"name": f"{rng.choice([x for x in STEMS if x != stem])} "
                             f"{rng.choice(SUFFIX)}",
                     "country": country,
                     "founded": rec["founded"],
                     "website": f"www.other{len(records)}.com"}
            pairs.append({"id": f"p{i}", "a": rec, "b": other, "gold": 0})
        else:                                 # unrelated company
            other = {"name": f"{rng.choice([x for x in STEMS if x != stem])} "
                             f"{rng.choice(SUFFIX)}",
                     "country": rng.choice([c for c in COUNTRIES if c != country]),
                     "founded": str(rng.randint(1950, 2015)),
                     "website": f"www.other{len(records)}.com"}
            pairs.append({"id": f"p{i}", "a": rec, "b": other, "gold": 0})
    return pairs


class DeterministicMatcher:
    """Stands in for the host model. Replace with your LLM call."""

    WEIGHTS = {"name": 0.45, "country": 0.25, "founded": 0.15, "website": 0.15}

    def __call__(self, request):
        a, b = request["es_context"], request["et_context"]
        score, evidence = 0.0, []
        for attr, w in self.WEIGHTS.items():
            va, vb = str(a.get(attr, "")), str(b.get(attr, ""))
            if not va or not vb:
                continue
            if va.lower() == vb.lower():
                score += w
                evidence.append(f"{attr}=same({va})")
            else:
                evidence.append(f"{attr}=differ({va} vs {vb})")

        label = 1 if score >= 0.50 else 0
        rationale = ("[DECISIVE] " + " ".join(evidence[:2]) + "\n"
                     "[SUPPORTING] " + " ".join(evidence[2:]))

        # A host model that can attribute its decision to rules lets Eq. 8
        # accumulate evidence; rules that agree with the verdict score 1.
        feedback = {r["rule_id"]: 1.0 for r in request["triggered_rules"]
                    if r["conclusion"] == label}
        return {"decision": label, "rationale": rationale,
                "rule_feedback": feedback,
                "confidence": min(0.99, 0.5 + 0.4 * abs(score - 0.5)),
                "token_usage": 380 + 7 * len(evidence)}


def evaluate(port, pairs, learn):
    labels, golds, records = [], [], []
    for p in pairs:
        out = port.decide(p["id"], p["id"] + "-b", p["a"], p["b"],
                          gold=p["gold"], learn=learn)
        labels.append(out.label)
        golds.append(p["gold"])
        records.append(out)
    scores = precision_recall_f1(golds, labels)
    eff = efficiency_summary([r.to_dict() for r in records])
    eff["llm_calls"] = sum(1 for r in records if r.stage == "stage2")
    return {"f1": scores["f1"], "precision": scores["precision"],
            "recall": scores["recall"], "efficiency": eff}


def main():
    pairs = build_stream()
    train, test = pairs[:80], pairs[80:]

    print("=" * 68)
    print("EvoRM offline demo - 80 training pairs, 40 held-out pairs")
    print("=" * 68)
    print(f"rule bank before learning: {EvoRMPort(config=CFG).stats()['hypergraph']['num_rules']} rules")

    cold = EvoRMPort(config=CFG, base_matcher=DeterministicMatcher(), verbose=False)
    cold_res = evaluate(cold, test, learn=False)

    warm = EvoRMPort(config=CFG, base_matcher=DeterministicMatcher(), verbose=False)
    for p in train:
        warm.decide(p["id"], p["id"] + "-b", p["a"], p["b"],
                    gold=p["gold"], learn=True)
    warm_res = evaluate(warm, test, learn=True)

    # Same records, but only name and country are populated: no learned rule
    # can fire, so every pair must fall through to the host model rather than
    # take a stale symbolic shortcut.
    shifted = []
    for p in test[:20]:
        keep = {"name", "country"}
        shifted.append({"id": p["id"] + "-s", "gold": p["gold"],
                        "a": {k: v for k, v in p["a"].items() if k in keep},
                        "b": {k: v for k, v in p["b"].items() if k in keep}})
    shifted_res = evaluate(warm, shifted, learn=False)

    print()
    print(f"{'':22}{'F1':>8}{'LLM calls':>12}{'tokens/q':>11}{'routed':>9}")
    print("-" * 68)
    for name, res in (("cold (empty bank)", cold_res),
                      ("warm (learned bank)", warm_res),
                      ("warm, 2 attrs only", shifted_res)):
        e = res["efficiency"]
        print(f"{name:22}{res['f1']:>8.3f}{e['llm_calls']:>12}"
              f"{e['avg_tokens']:>11.1f}{e['routing_rate']:>9.1%}")

    stats = warm.stats()
    print()
    print(f"rules stored        : {stats['hypergraph']['num_rules']}")
    print(f"hyperedges          : {stats['hypergraph']['num_hyperedges']}")
    print(f"mean rule confidence: {stats['maintenance']['mean_conf']:.3f}")

    print()
    print("Top rules after the stream:")
    bank = list(warm.plugin.hypergraph.rules.values())
    for rule in sorted(bank, key=lambda r: r.conf, reverse=True)[:5]:
        body = " AND ".join(f"{a.atom_type}({a.attr})" for a in rule.atoms)
        verdict = "MATCH" if rule.conclusion == 1 else "NON-MATCH"
        print(f"  conf={rule.conf:.3f}  triggers={rule.trigger_count:<3} "
              f"{body} -> {verdict}")

    # ------------------------------------------------------------------ rerank
    print()
    print("rerank(): the EA / EL entry point (base score + symbolic evidence)")
    query = {"name": "Acme Inc", "country": "US",
             "founded": "1999", "website": "www.acme.com"}
    cands = [
        ("c1", {"name": "Acme Inc", "country": "US", "founded": "1999",
                "website": "www.acme.com"}),
        ("c2", {"name": "Acme Inc", "country": "US", "founded": "2000",
                "website": "www.acme.com"}),
        ("c3", {"name": "Acme GmbH", "country": "DE", "founded": "1999",
                "website": "www.acme.de"}),
        ("c4", {"name": "Globex", "country": "US", "founded": "1999",
                "website": "www.globex.com"}),
        ("c5", {"name": "Initech", "country": "US", "founded": "2001",
                "website": "www.initech.com"}),
    ]
    base = {"c1": 0.71, "c2": 0.69, "c3": 0.66, "c4": 0.70, "c5": 0.65}
    print(f"  {'cand':6}{'base':>7}{'evidence':>10}{'final':>8}")
    for r in warm.rerank("q1", query, cands, base):
        print(f"  {r.entity_id:6}{r.base_score:>7.2f}{r.evidence:>10.2f}"
              f"{r.final_score:>8.2f}")

    # Persistence round-trip
    import shutil
    import tempfile
    tmp = tempfile.mkdtemp(prefix="evorm-demo-")
    try:
        warm.save(tmp)
        reloaded = EvoRMPort(config=CFG, base_matcher=DeterministicMatcher())
        ok = reloaded.load(tmp)
        print()
        print(f"save/load round-trip: {'ok' if ok else 'FAILED'}, "
              f"{reloaded.stats()['hypergraph']['num_rules']} rules restored")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("=" * 68)


if __name__ == "__main__":
    main()