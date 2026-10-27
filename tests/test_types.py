"""Data structures and the equations attached to them (Eq. 6, 8, 9)."""

import math
import time

import pytest

from evorm_core.types import ConditionAtom, FOLRule, Hyperedge


def _cos(a, b):
    return float(sum(x * y for x, y in zip(a, b)) / (
        math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))))


def test_eq6_weight_uses_per_pair_cosine():
    """Eq. 6: w_k = alpha * max Conf + beta * mean over ENTITY PAIRS of cos."""
    rule = FOLRule(rule_id="r", atoms=[ConditionAtom("SameValue", "name")], conclusion=1, conf=0.8)
    edge = Hyperedge(hyperedge_id="h", entity_pairs={(1, 2), (3, 4)}, rules=[rule])
    embs = {"e_1": [1.0, 0.0], "e_2": [1.0, 0.0],
            "e_3": [1.0, 0.0], "e_4": [0.0, 1.0]}

    w = edge.recompute_weight(alpha=0.6, beta=0.4, entity_embeddings=embs, similarity_fn=_cos)

    # pair (1,2) -> 1.0 ; pair (3,4) -> 0.0 ; mean = 0.5
    assert w == pytest.approx(0.6 * 0.8 + 0.4 * 0.5)


def test_eq6_without_embeddings_degrades_to_symbolic():
    rule = FOLRule(rule_id="r", atoms=[ConditionAtom("SameValue", "name")], conclusion=1, conf=0.5)
    edge = Hyperedge(hyperedge_id="h", entity_pairs={(1, 2)}, rules=[rule])
    assert edge.recompute_weight(0.6, 0.4, None) == pytest.approx(0.3)


def test_eq8_confidence():
    rule = FOLRule(rule_id="r", atoms=[], conclusion=1, conf0=0.4, sR_sum=0.6, trigger_count=2)
    rule.update_confidence()
    assert rule.conf == pytest.approx((0.4 + 0.6) / 3)


def test_eq9_freshness_decays():
    rule = FOLRule(rule_id="r", atoms=[], conclusion=1, last_used_time=time.time() - 100)
    rule.update_freshness(decay_lambda=0.01)
    assert rule.fresh == pytest.approx(math.exp(-1.0), abs=1e-3)

    fresh_rule = FOLRule(rule_id="r2", atoms=[], conclusion=1)
    fresh_rule.update_freshness(0.01)
    assert fresh_rule.fresh == pytest.approx(1.0)


def test_serialization_roundtrip():
    rule = FOLRule(rule_id="r1", atoms=[ConditionAtom("SemanticEquiv", "venue", threshold=0.85)],
                   conclusion=1, conf=0.7, conf0=0.5, trigger_count=3)
    clone = FOLRule.from_dict(rule.to_dict())
    assert clone.rule_id == rule.rule_id
    assert clone.atoms[0].threshold == pytest.approx(0.85)
    assert clone.conf == pytest.approx(0.7)
