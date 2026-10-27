"""Two-stage controller: routing conditions of Section III-E."""

import pytest

from evorm_core.controller import TwoStageInferenceController
from evorm_core.hypergraph import HypergraphStorage
from evorm_core.types import ConditionAtom, FOLRule

A = {"name": "Acme Corp", "country": "US"}
B = {"name": "Acme Corp", "country": "US"}
C = {"name": "Globex", "country": "US"}


def _setup(rules):
    hg = HypergraphStorage()
    for rule in rules:
        hg.add_trajectory((1, 2), {"e_1", "e_2", "a_name"}, rule)
    return hg, TwoStageInferenceController(hg, theta_hi=0.65, theta_prune=0.75)


def _rule(rid, conclusion, conf, attr="name", trig=0):
    return FOLRule(rule_id=rid, atoms=[ConditionAtom("SameValue", attr)],
                   conclusion=conclusion, conf=conf, trigger_count=trig)


def test_direct_matching_above_theta_hi():
    hg, ctrl = _setup([_rule("r1", 1, 0.9)])
    decision, triggered, _ = ctrl.stage1_symbolic_filtering("1", "2", A, B)
    assert decision == 1
    assert len(triggered) == 1
    assert ctrl.stage1_match_hits == 1


def test_below_threshold_survives():
    hg, ctrl = _setup([_rule("r1", 1, 0.50)])
    decision, _, _ = ctrl.stage1_symbolic_filtering("1", "2", A, B)
    assert decision is None


def test_conflicting_verdicts_survive():
    hg, ctrl = _setup([_rule("r1", 1, 0.9), _rule("r2", 0, 0.9)])
    decision, triggered, _ = ctrl.stage1_symbolic_filtering("1", "2", A, B)
    assert decision is None
    assert len(triggered) == 2


def test_direct_non_matching_above_theta_prune():
    hg, ctrl = _setup([_rule("r1", 0, 0.80, attr="country")])
    # atom SameValue(country) holds for A/B -> the rule triggers, verdict 0
    decision, _, _ = ctrl.stage1_symbolic_filtering("1", "2", A, B)
    assert decision == 0
    assert ctrl.stage1_nonmatch_hits == 1


def test_rule_not_triggered_when_atom_fails():
    hg, ctrl = _setup([_rule("r1", 1, 0.9)])
    decision, triggered, _ = ctrl.stage1_symbolic_filtering("1", "2", A, C)
    assert decision is None
    assert triggered == []


def test_stage1_does_not_mutate_confidence():
    """The controller must not apply any extra confidence boost."""
    rule = _rule("r1", 1, 0.9)
    hg, ctrl = _setup([rule])
    before = rule.conf
    ctrl.stage1_symbolic_filtering("1", "2", A, B)
    assert rule.conf == pytest.approx(before)


def test_eq8_trigger_count_counts_stage1_route():
    rule = _rule("r1", 1, 0.9, trig=0)
    hg, ctrl = _setup([rule])
    decision, triggered, _ = ctrl.stage1_symbolic_filtering("1", "2", A, B)
    ctrl.record_trajectory(triggered, decisive_rule=triggered[0])

    assert rule.trigger_count == 1          # denominator of Eq. 8
    assert rule.sR_sum == pytest.approx(1.0)  # stage-1 route is decisive


class _RecordingHeavy:
    """Stands in for M_heavy; records the order in which it is called."""

    def __init__(self, log):
        self.log = log

    def decide(self, es_context, et_context, rationale=None,
               triggered_rules=None, base_prompt=None):
        self.log.append("heavy")
        return {"decision": 1, "confidence": 0.9, "rule_feedback": {},
                "summary": "same organisation", "raw_response": "",
                "token_usage": 11}


class _RecordingLight:
    """Stands in for M_light; records the decision it was asked to explain."""

    def __init__(self, log):
        self.log = log
        self.seen_decision = "unset"

    def elicit(self, es_context, et_context, triggered_rules=None,
               custom_prompt=None, decision=None):
        from evorm_core.mlight import RationaleResult
        self.log.append("light")
        self.seen_decision = decision
        return RationaleResult(
            rationale="[DECISIVE] identical names and countries", token_usage=7)


def test_stage2_runs_mheavy_before_mlight():
    """Section III-C: y_hat = M_heavy(...) is computed before r = M_light(..., y_hat)."""
    hg, ctrl = _setup([_rule("r1", 1, 0.5)])
    log = []
    ctrl.mheavy = _RecordingHeavy(log)
    ctrl.mlight = _RecordingLight(log)
    ctrl.use_mlight = True

    out = ctrl.stage2_llm_judgment("1", "2", A, B, [], [])

    assert log == ["heavy", "light"]
    assert ctrl.mlight.seen_decision == 1
    assert out["decision"] == 1
    assert out["token_usage"] == 18
    assert out["method"] == "mheavy-only"


def test_stage2_skips_mlight_when_mheavy_has_no_client():
    hg, ctrl = _setup([_rule("r1", 1, 0.5)])
    log = []

    class _SilentHeavy:
        def decide(self, *a, **kw):
            log.append("heavy")
            return {"decision": None, "confidence": 0.5, "rule_feedback": {},
                    "summary": "no client available", "raw_response": "",
                    "token_usage": 0}

    ctrl.mheavy = _SilentHeavy()
    ctrl.mlight = _RecordingLight(log)
    ctrl.use_mlight = True

    out = ctrl.stage2_llm_judgment("1", "2", A, B, [], [])

    assert log == ["heavy"]
    assert out["decision"] is None
    assert ctrl.stage2_hits == 0
