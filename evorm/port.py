#!/usr/bin/env python3
"""
EvoRM port -- the single integration surface for existing matching frameworks.

EvoRM does not own your model and does not compute your metrics.  It exposes
three calls; everything else (Hits@1 / MRR / F1 / accuracy, dataset loading,
retrieval) stays in your framework.

    decide(...)   binary decision for one pair         -> ER, SM, EA confirmation
    rerank(...)   evidence-aware rescoring of a list    -> EA, EL
    observe(...)  feed an external decision back        -> rule learning only

Typical EA / EL integration (you already have candidates and scores):

    from evorm.port import EvoRMPort

    port = EvoRMPort()
    for query, gold in test_set:
        ranks = port.rerank(query.id, query.context, candidates, base_scores)
        predicted = ranks[0].entity_id
        # compute Hits@1 / Hits@10 / MRR with YOUR existing evaluator

Typical ER integration (EvoRM makes the decision, your LLM answers):

    def my_matcher(request) -> dict:
        # request["prompt"], request["es_context"], request["triggered_rules"]
        return {"decision": 1, "rationale": my_llm(request), "token_usage": 812}

    port = EvoRMPort(base_matcher=my_matcher)
    out = port.decide("r1", "r2", rec_a, rec_b)
    # out.label in {0, 1}

Typical "augment a framework that already decides" integration:

    port.observe(qid, cid, q_context, c_context,
                 decision=fw.last_decision, rationale=fw.last_rationale)
    # EvoRM distils rules from the rationale; no LLM call is made here.
"""

import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from evorm_core.config import EvoRMConfig
from evorm_core.plugin import EvoRMPlugin


@dataclass
class Decision:
    """Outcome of one EvoRM `decide` call."""
    label: int
    confidence: float
    stage: str                       # "stage1" | "stage2"
    rationale: str = ""
    tokens: int = 0
    latency_s: float = 0.0
    triggered_rules: List[str] = field(default_factory=list)
    new_rule: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ScoredCandidate:
    """One rescored candidate returned by `rerank`."""
    entity_id: str
    base_score: float
    evidence: float                  # symbolic evidence in [-1, 1]
    final_score: float


class EvoRMPort:
    """Single integration point; wraps EvoRMPlugin with a framework-friendly API."""

    def __init__(self,
                 config: Optional[EvoRMConfig] = None,
                 client=None,
                 mlight_client=None,
                 base_matcher: Optional[Callable] = None,
                 evidence_weight: float = 1.0,
                 verbose: bool = False,
                 embedding_fn: Optional[Callable] = None,
                 **plugin_kwargs):
        self.plugin = EvoRMPlugin(
            config=config, client=client, mlight_client=mlight_client,
            verbose=verbose, embedding_fn=embedding_fn, **plugin_kwargs)
        self.base_matcher = base_matcher
        self.evidence_weight = evidence_weight

    # ------------------------------------------------------------------ decide
    def decide(self, es_id: str, et_id: str,
               es_context: Dict, et_context: Dict,
               prompt: str = "", gold: Optional[int] = None,
               learn: bool = True) -> Decision:
        """Return a binary match decision for one entity pair."""
        out = self.plugin.process(
            es_id, et_id, es_context, et_context,
            base_prompt=prompt, base_matcher=self.base_matcher,
            gold=gold, learn=learn)

        label = out.get("decision")
        if label is None:                      # the host model abstained
            label = 0
        return Decision(
            label=int(label),
            confidence=float(out.get("confidence", 0.0)),
            stage=out.get("stage", "stage2"),
            rationale=out.get("rationale", ""),
            tokens=int(out.get("tokens", 0)),
            latency_s=float(out.get("latency_s", 0.0)),
            triggered_rules=out.get("triggered_rules", []),
            new_rule=out.get("new_rule"),
        )

    # ------------------------------------------------------------------ score
    def score(self, es_id: str, et_id: str,
              es_context: Dict, et_context: Dict) -> float:
        """Symbolic evidence for a pair, in [-1, 1]. Does not call any LLM."""
        triggered, _ = self.plugin.controller.trigger_rules(
            es_id, et_id, es_context, et_context)
        if not triggered:
            return 0.0

        pos = [r.conf for r in triggered if r.conclusion == 1]
        neg = [r.conf for r in triggered if r.conclusion == 0]

        if pos and not neg:
            return min(1.0, max(pos))
        if neg and not pos:
            return -min(1.0, max(neg))
        # conflicting verdicts: confidence-weighted margin
        return min(1.0, max(pos, default=0.0)) - min(1.0, max(neg, default=0.0))

    # ------------------------------------------------------------------ rerank
    def rerank(self, query_id: str, query_context: Dict,
               candidates: Sequence,
               base_scores: Optional[Dict[str, float]] = None) -> List[ScoredCandidate]:
        """Rescore and sort candidates for EA / EL pipelines.

        ``candidates`` may be ``(entity_id, context)`` tuples or objects with
        ``.entity_id`` / ``.context`` attributes. ``base_scores`` maps candidate
        id -> the host model's score; missing entries default to 0.0.
        """
        base_scores = base_scores or {}
        scored: List[ScoredCandidate] = []

        for cand in candidates:
            if isinstance(cand, (tuple, list)):
                cid, cctx = cand[0], cand[1]
            else:
                cid, cctx = cand.entity_id, cand.context

            base = float(base_scores.get(str(cid), 0.0))
            evidence = self.score(str(query_id), str(cid), query_context, cctx or {})
            scored.append(ScoredCandidate(
                entity_id=str(cid), base_score=base, evidence=evidence,
                final_score=base + self.evidence_weight * evidence))

        scored.sort(key=lambda s: s.final_score, reverse=True)
        return scored

    # ------------------------------------------------------------------ observe
    def observe(self, es_id: str, et_id: str,
                es_context: Dict, et_context: Dict,
                decision: int, rationale: str = "",
                rule_feedback: Optional[Dict[str, float]] = None,
                tokens: int = 0, prompt: str = "") -> Optional[str]:
        """Learn from a decision your framework already made.

        No LLM is invoked. EvoRM encodes the rationale into a FOL rule, stores
        it in the hypergraph, and updates the statistics of triggered rules.
        Returns the id of the newly created rule, if any.

        ``rule_feedback`` maps rule id -> contribution in [0, 1]. Frameworks
        that do not score individual rules can leave it out: the rules whose
        conclusion agrees with ``decision`` then count as decisive.
        """
        plugin = self.plugin
        plugin._remember_entities(es_id, et_id, es_context, et_context)

        triggered, _ = plugin.controller.trigger_rules(
            es_id, et_id, es_context, et_context)

        new_rule_id = None
        if rationale:
            rule = plugin.rule_encoding.create_rule(
                rationale=rationale,
                conclusion=int(decision),
                es_context=es_context,
                et_context=et_context,
                used_rules=[(r, r.conf) for r in triggered])
            if rule is not None:
                plugin.hypergraph.add_trajectory(
                    (es_id, et_id),
                    plugin._node_set(es_id, et_id, rule),
                    rule)
                new_rule_id = rule.rule_id

        feedback = rule_feedback
        if feedback is None:
            feedback = {r.rule_id: 1.0 for r in triggered
                        if r.conclusion == int(decision)}
        plugin.controller.record_trajectory(triggered, rule_feedback=feedback)
        plugin.query_count += 1
        plugin.total_tokens += int(tokens)
        plugin._maybe_optimize()
        return new_rule_id

    # ------------------------------------------------------------------ lifecycle
    def save(self, directory: str):
        self.plugin.save(directory)

    def load(self, directory: str) -> bool:
        return self.plugin.load(directory)

    def stats(self) -> Dict:
        stats = self.plugin.stats()
        total = max(1, self.plugin.query_count)
        stats["routing_rate"] = stats["controller"]["stage1_rate"]
        stats["avg_latency_s"] = 0.0
        stats["queries"] = self.plugin.query_count
        stats["avg_tokens_per_query"] = self.plugin.total_tokens / total
        return stats

    def reset_stats(self):
        c = self.plugin.controller
        c.stage1_total = c.stage1_hits = 0
        c.stage1_match_hits = c.stage1_nonmatch_hits = c.stage1_gate_hits = 0
        c.stage2_total = c.stage2_hits = c.trajectory_total = 0
        self.plugin.query_count = 0
        self.plugin.total_tokens = 0


class EvoRMLearner:
    """Convenience wrapper for the "observe only" mode (no owned LLM).

    Use this when your framework (ChatEA, MatchGPT, LELA, AdacoAgentEA, ...)
    already produces decisions and rationales: EvoRM becomes a pure rule-memory
    layer that improves retrieval evidence without touching your model calls.
    """

    def __init__(self, port: Optional[EvoRMPort] = None, **kwargs):
        self.port = port or EvoRMPort(**kwargs)

    def ingest(self, trajectories: Sequence[Dict]) -> int:
        """Ingest a batch of ``{es_id, et_id, es_context, et_context,
        decision, rationale, tokens}`` records."""
        n = 0
        for t in trajectories:
            self.port.observe(
                t["es_id"], t["et_id"], t["es_context"], t["et_context"],
                decision=int(t["decision"]),
                rationale=t.get("rationale", ""),
                rule_feedback=t.get("rule_feedback"),
                tokens=int(t.get("tokens", 0)))
            n += 1
        return n

    def score(self, *args, **kwargs) -> float:
        return self.port.score(*args, **kwargs)

    def rerank(self, *args, **kwargs) -> List[ScoredCandidate]:
        return self.port.rerank(*args, **kwargs)

    def stats(self) -> Dict:
        return self.port.stats()