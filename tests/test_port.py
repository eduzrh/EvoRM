"""Integration port: decide / score / rerank / observe."""

import pytest

from evorm.port import EvoRMLearner, EvoRMPort
from evorm_core.config import EvoRMConfig


class _Msg:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Msg(content)


class _Usage:
    total_tokens = 100


class _Response:
    def __init__(self, content):
        self.choices = [_Choice(content)]
        self.usage = _Usage()


class _Completions:
    def create(self, **kwargs):
        prompt = str(kwargs["messages"][0]["content"]).lower()
        if "rationale" in prompt and "decision" not in prompt:
            return _Response("[DECISIVE] name=same\n[SUPPORTING] country=same")
        return _Response('{"decision": "MATCH", "rule_feedback": {"rule_1": 0.9}, '
                         '"reasoning": "[DECISIVE] name=same"}')


class _Chat:
    completions = _Completions()


class MockClient:
    chat = _Chat()


A = {"name": "Acme Corp", "country": "US"}
B = {"name": "Acme Corp", "country": "US"}
Z = {"name": "Globex", "country": "DE"}


def _port(**kwargs):
    cfg = EvoRMConfig(enable_mlp_gate=False, **kwargs)
    return EvoRMPort(config=cfg, client=MockClient(), mlight_client=MockClient())


def test_decide_returns_binary_label_and_tokens():
    port = _port()
    out = port.decide("a", "b", A, B)
    assert out.label in (0, 1)
    assert out.tokens > 0
    assert out.stage == "stage2"


def test_learning_enables_stage1_and_zero_tokens():
    port = _port()
    for i in range(10):
        out = port.decide(f"a{i}", f"b{i}", A, B)

    assert out.label == 1
    assert port.stats()["routing_rate"] > 0
    assert out.tokens == 0 or out.stage == "stage2"


def test_score_is_positive_for_matching_pair():
    port = _port()
    for i in range(6):
        port.decide(f"a{i}", f"b{i}", A, B)
    assert port.score("a99", "b99", A, B) > 0


def test_rerank_prefers_the_supported_candidate():
    port = _port()
    for i in range(6):
        port.decide(f"a{i}", f"b{i}", A, B)

    ranked = port.rerank("q", A, [("z", Z), ("b", B)],
                         base_scores={"z": 0.5, "b": 0.5})
    assert ranked[0].entity_id == "b"
    assert ranked[0].final_score >= ranked[1].final_score


def test_observe_learns_without_any_llm_call():
    port = _port()
    rid = port.observe("a", "b", A, B, decision=1,
                       rationale="[DECISIVE] name=same")
    assert rid is not None
    assert port.plugin.hypergraph.stats()["num_rules"] == 1
    assert port.plugin.total_tokens == 0


def test_learner_batch_ingest():
    learner = EvoRMLearner(port=_port())
    n = learner.ingest([
        {"es_id": "a1", "et_id": "b1", "es_context": A, "et_context": B,
         "decision": 1, "rationale": "[DECISIVE] name=same"},
        {"es_id": "a2", "et_id": "b2", "es_context": A, "et_context": B,
         "decision": 1, "rationale": "[DECISIVE] name=same"},
    ])
    assert n == 2
    assert learner.stats()["queries"] == 2
