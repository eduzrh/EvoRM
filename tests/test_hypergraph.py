"""Hypergraph storage: Eq. 7 merging, O(1) retrieval, flat ablation mode."""

import pytest

from evorm_core.hypergraph import HypergraphStorage
from evorm_core.types import ConditionAtom, FOLRule


def _rule(rid, attr="name", conf=0.5, conclusion=1, trig=0):
    return FOLRule(rule_id=rid, atoms=[ConditionAtom("SameValue", attr)],
                   conclusion=conclusion, conf=conf, trigger_count=trig)


def test_eq7_merges_overlapping_nodes():
    hg = HypergraphStorage(merge_threshold=0.5)
    hg.add_trajectory((1, 1), {"e_1", "e_2", "a_name"}, _rule("r1"))
    hg.add_trajectory((1, 2), {"e_1", "e_3", "a_name"}, _rule("r2"))

    # Jaccard = |{e_1, a_name}| / |{e_1,e_2,e_3,a_name}| = 0.5 -> merges
    assert len(hg.hyperedges) == 1
    edge = next(iter(hg.hyperedges.values()))
    assert edge.entity_pairs == {(1, 1), (1, 2)}
    assert {r.rule_id for r in edge.rules} == {"r1", "r2"}


def test_eq7_keeps_disjoint_nodes_separate():
    hg = HypergraphStorage(merge_threshold=0.5)
    hg.add_trajectory((1, 1), {"e_1", "e_2", "a_name"}, _rule("r1"))
    hg.add_trajectory((9, 9), {"e_9", "e_8", "a_price"}, _rule("r2", attr="price"))
    assert len(hg.hyperedges) == 2


def test_retrieval_by_attribute_fallback():
    hg = HypergraphStorage()
    hg.add_trajectory((1, 1), {"e_1", "e_2", "a_name"}, _rule("r1"))

    # entity ids unseen -> resolved through the attribute index
    rules = hg.get_candidate_rules(["77", "88"], [{"name": "a"}, {"name": "b"}])
    assert [r.rule_id for r in rules] == ["r1"]


def test_flat_mode_returns_all_rules():
    hg = HypergraphStorage(flat_mode=True)
    hg.add_trajectory((1, 1), {"e_1"}, _rule("r1"))
    hg.add_trajectory((9, 9), {"e_9"}, _rule("r2"))

    assert len(hg.get_candidate_rules(["123"], [{}])) == 2
    assert hg.stats()["num_rules"] == 2


def test_remove_rule_cleans_hyperedges():
    hg = HypergraphStorage()
    hg.add_trajectory((1, 1), {"e_1", "e_2", "a_name"}, _rule("r1"))
    hg.remove_rule("r1")
    assert hg.rules == {}
    assert all(not he.rules for he in hg.hyperedges.values())


def test_serialization_roundtrip():
    hg = HypergraphStorage()
    hg.add_trajectory((1, 1), {"e_1", "e_2", "a_name"}, _rule("r1", conf=0.7))
    clone = HypergraphStorage.from_dict(hg.to_dict())

    assert clone.stats()["num_rules"] == hg.stats()["num_rules"]
    assert clone.stats()["num_hyperedges"] == hg.stats()["num_hyperedges"]
    assert clone.get_rule("r1").conf == pytest.approx(0.7)
