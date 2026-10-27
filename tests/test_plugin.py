"""Node construction and the offline learning loop."""

from evorm.port import EvoRMPort
from evorm_core.config import EvoRMConfig
from evorm_core.plugin import EvoRMPlugin
from evorm_core.types import ConditionAtom, FOLRule

CFG = EvoRMConfig(enable_mlp_gate=False)

A = {"name": "Acme Inc", "country": "US", "founded": "1999"}
A2 = {"name": "Acme Inc", "country": "US", "founded": "1999"}
B = {"name": "Globex", "country": "DE", "founded": "2001"}


def _rule(atoms):
    return FOLRule(rule_id="r1", atoms=atoms, conclusion=1)


def test_node_set_spans_entity_attribute_value_and_neighbour_nodes():
    """V = V_E u V_A u V_V, plus graph-dependent nodes (Section III-D)."""
    plugin = EvoRMPlugin(config=CFG)
    rule = _rule([
        ConditionAtom("SameValue", "name", value1="Acme Inc"),
        ConditionAtom("DifferValue", "country", value1="US", value2="DE"),
        ConditionAtom("ShareNeighbor", "industry"),
    ])

    nodes = plugin._node_set("1", "2", rule)

    assert {"e_1", "e_2"} <= nodes                 # V_E
    assert {"a_name", "a_country", "a_industry"} <= nodes   # V_A
    assert "v_name=acme inc" in nodes              # V_V
    assert "n_industry" in nodes                   # graph-dependent
    assert not any(n.startswith("v_country") for n in nodes)


def test_observe_then_decide_routes_without_a_model():
    """A rule learned from an external decision answers the next identical pair."""
    port = EvoRMPort(config=CFG)

    rationale = ("[DECISIVE] name=same(Acme Inc) country=same(US)\n"
                 "[SUPPORTING] founded=same(1999)")
    for _ in range(3):
        port.observe("a", "b", A, A2, decision=1, rationale=rationale)

    out = port.decide("a2", "b2", A, A2, learn=False)

    assert out.stage == "stage1"
    assert out.label == 1
    assert out.tokens == 0
    assert out.triggered_rules


def test_negative_evidence_demotes_an_unrelated_candidate():
    port = EvoRMPort(config=CFG)
    rationale = ("[DECISIVE] name=differ(Acme Inc vs Globex) "
                 "country=differ(US vs DE)\n[SUPPORTING] none")
    for _ in range(3):
        port.observe("a", "b", A, B, decision=0, rationale=rationale)

    assert port.score("a", "b", A, B) < 0