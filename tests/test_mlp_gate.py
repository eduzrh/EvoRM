"""MLP gate: 64-dim feature layout of Section III-E."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from evorm_core.mlp_gate import EMB_DIM, FEATURE_DIM, MLPGate, RULE_DIM, TOPO_DIM
from evorm_core.types import ConditionAtom, FOLRule


def test_feature_layout_matches_paper_groups():
    assert FEATURE_DIM == 64
    assert EMB_DIM + RULE_DIM + TOPO_DIM == 64


def test_embedding_block_is_populated():
    gate = MLPGate(n_warmup=10)
    rule = FOLRule(rule_id="r", atoms=[ConditionAtom("SameValue", "name")],
                   conclusion=1, conf=0.8, trigger_count=3)
    emb = np.ones(1024, dtype=np.float32)

    feats = gate.extract_features({"name": "a"}, {"name": "a"}, [rule],
                                  es_embedding=emb, et_embedding=emb)
    assert feats.shape == (FEATURE_DIM,)
    assert feats[0] == pytest.approx(1.0)      # cosine of identical vectors


def test_gate_skips_when_untrained():
    gate = MLPGate(n_warmup=10)
    feats = np.zeros(FEATURE_DIM, dtype=np.float32)
    assert gate.should_invoke_llm(feats) is True


def test_gate_trains_and_predicts():
    gate = MLPGate(n_warmup=20, theta_gate=0.5)
    rng = np.random.RandomState(0)
    for _ in range(40):
        x = rng.normal(0, 0.1, FEATURE_DIM).astype(np.float32)
        gate.collect_sample(x, 1.0)
        gate.collect_sample(x + 5.0, 0.0)

    assert gate.is_ready_to_train()
    gate.fit_model(epochs=10, verbose=False)
    assert gate.is_trained
    assert gate.predict_proba(np.full(FEATURE_DIM, 5.0, dtype=np.float32)) < 0.5
