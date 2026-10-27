"""Rule maintenance: pruning, flipping, merging (Eq. 10), forgetting."""

import time

import pytest

from evorm_core.hypergraph import HypergraphStorage
from evorm_core.maintenance import RuleMaintenance
from evorm_core.types import ConditionAtom, FOLRule


def _rule(rid, atoms, conclusion=1, conf=0.5, trig=0, used=0):
    return FOLRule(rule_id=rid, atoms=atoms, conclusion=conclusion,
                   conf=conf, conf0=conf, trigger_count=trig, used_count=used)


def _hg(*rules):
    hg = HypergraphStorage()
    for r in rules:
        hg.rules[r.rule_id] = r
        hg.add_trajectory((1, 2), {"e_1", "e_2"}, r)
    return hg


def test_prunes_stale_rules():
    stale = _rule("stale", [ConditionAtom("SameValue", "name")])
    stale.last_used_time = time.time() - 10_000
    fresh = _rule("fresh", [ConditionAtom("SameValue", "price")])
    hg = _hg(stale, fresh)

    summary = RuleMaintenance(hg, decay_lambda=0.01, freshness_threshold=0.1).optimize()
    assert summary["pruned"] == 1
    assert set(hg.rules) == {"fresh"}


def test_flips_low_confidence_positive_rule():
    rule = _rule("r1", [ConditionAtom("SameValue", "name")], conclusion=1,
                 conf=0.1, trig=10, used=0)
    hg = _hg(rule)

    summary = RuleMaintenance(hg, confidence_threshold=0.2, eval_triggers=10).optimize()
    assert summary["flipped"] == 1
    assert rule.conclusion == 0
    assert rule.conf == pytest.approx(0.3)
    assert rule.trigger_count == 0
    assert rule.flipped_at_trigger == 0


def test_never_flips_negative_rules():
    rule = _rule("r1", [ConditionAtom("SameValue", "name")], conclusion=0,
                 conf=0.05, trig=20, used=0)
    hg = _hg(rule)
    RuleMaintenance(hg, confidence_threshold=0.2, eval_triggers=10).optimize()
    assert rule.conclusion == 0


def test_prunes_after_ineffective_flip():
    rule = _rule("r1", [ConditionAtom("SameValue", "name")], conclusion=1,
                 conf=0.1, trig=10, used=0)
    hg = _hg(rule)
    mt = RuleMaintenance(hg, confidence_threshold=0.2, eval_triggers=10)
    mt.optimize()                                  # flip
    assert "r1" in hg.rules

    rule.trigger_count = 10                        # N_eval additional triggers
    rule.conf = 0.1                                # still not recovering
    mt.optimize()
    assert "r1" not in hg.rules


def test_merging_uses_frequency_weighted_confidence():
    """Eq. 10: Conf(R_c) = sum(Conf_i * |Trig_i|) / sum(|Trig_i|)."""
    a = _rule("a", [ConditionAtom("SameValue", "name"),
                    ConditionAtom("SameValue", "type"),
                    ConditionAtom("DifferValue", "price")],
              conf=0.6, trig=10, used=5)
    b = _rule("b", [ConditionAtom("SameValue", "name"),
                    ConditionAtom("SameValue", "type"),
                    ConditionAtom("DifferValue", "price"),
                    ConditionAtom("ShareNeighbor", "country")],
              conf=0.8, trig=5, used=3)
    # Jaccard = 3/4 = 0.75 >= eta_m
    hg = _hg(a, b)

    summary = RuleMaintenance(hg, merge_similarity=0.7, meta_rule_fn=None).optimize()

    assert summary["merged"] == 1
    assert len(hg.rules) == 1
    meta = next(iter(hg.rules.values()))
    assert meta.conf == pytest.approx((0.6 * 10 + 0.8 * 5) / 15)
    assert meta.trigger_count == 15
    assert {a.to_key() for a in meta.atoms} == {
        "SameValue(name)", "SameValue(type)", "DifferValue(price)"}


def test_capacity_forgetting_evicts_lowest_utility():
    rules = [_rule(f"r{i}", [ConditionAtom("SameValue", f"a{i}")], conf=0.1 * (i + 1))
             for i in range(10)]
    hg = _hg(*rules)
    RuleMaintenance(hg, max_rules=5, evict_percentile=0.2).optimize()
    assert len(hg.rules) == 8            # evicted 20% of 10
    assert "r0" not in hg.rules          # lowest Conf * Fresh goes first
