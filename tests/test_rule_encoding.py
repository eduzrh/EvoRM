"""Rule encoding: Eq. 4 initial confidence and atom extraction."""

import pytest

from evorm_core.rule_encoding import RuleEncoding
from evorm_core.types import ConditionAtom, FOLRule


def test_eq4_zero_without_used_rules():
    enc = RuleEncoding(client=None)
    atoms = [ConditionAtom("SameValue", "name")]
    assert enc.initial_confidence(atoms, used_rules=None) == pytest.approx(0.0)
    assert enc.initial_confidence(atoms, used_rules=[]) == pytest.approx(0.0)


def test_eq4_takes_max_contribution_of_overlapping_rules():
    enc = RuleEncoding(client=None)
    atoms = [ConditionAtom("SameValue", "name"), ConditionAtom("DifferValue", "price")]
    other = FOLRule(rule_id="x", atoms=[ConditionAtom("SameValue", "name")], conclusion=1)

    conf0 = enc.initial_confidence(atoms, used_rules=[(other, 0.3), (other, 0.6)])
    assert conf0 == pytest.approx(0.6)


def test_eq4_no_floor_for_non_overlapping_rules():
    enc = RuleEncoding(client=None)
    atoms = [ConditionAtom("SameValue", "name")]
    disjoint = FOLRule(rule_id="y", atoms=[ConditionAtom("DifferNeighbor", "country")], conclusion=1)
    assert enc.initial_confidence(atoms, used_rules=[(disjoint, 1.0)]) == pytest.approx(0.0)


def test_template_parsing():
    enc = RuleEncoding(client=None)
    atoms = enc.parse_rationale("[DECISIVE] name=same\n[SUPPORTING] type=same", conclusion=1)
    keys = {a.to_key() for a in atoms}
    assert "SameValue(name)" in keys
    assert "SameValue(type)" in keys


def test_semantic_atoms_carry_thresholds():
    enc = RuleEncoding(client=None)
    atoms = enc.parse_semantic_atoms([("venue", 0.85, "")], [])
    assert atoms and atoms[0].atom_type == "SemanticEquiv"
    assert atoms[0].threshold == pytest.approx(0.85)


def test_rule_id_is_deterministic():
    enc = RuleEncoding(client=None)
    a = [ConditionAtom("SameValue", "name"), ConditionAtom("DifferValue", "price")]
    b = list(reversed(a))
    assert enc.stable_rule_id(a) == enc.stable_rule_id(b)
