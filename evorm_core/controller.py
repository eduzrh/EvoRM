#!/usr/bin/env python3
"""
Two-Stage Inference Controller -- paper Section III-E.

Stage 1 (symbolic, zero LLM cost):
    Direct matching     : all triggered rules predict 1 and max Conf >= theta_hi
    Direct non-matching : all triggered rules predict 0 and max Conf >= theta_prune
    Survival            : otherwise, forwarded to the MLP gate

MLP gate g_phi:
    A surviving pair enters Stage 2 only if g_phi(x_st) >= theta_gate;
    otherwise it is directly assigned a negative decision.

Stage 2 (LLM):
    Mheavy decides first and returns per-rule contribution scores s_R in
    [0, 1] (these feed Eq. 8); Mlight then explains the decision and its
    rationale is what RuleEncoding converts into a new rule.
"""

import json
import re
from typing import Callable, Dict, List, Optional, Tuple

from .types import ConditionAtom, FOLRule


class TwoStageInferenceController:
    """Symbolic filtering + gated LLM judgment with rule-level feedback."""

    def __init__(
        self,
        hypergraph,
        client=None,
        theta_hi: float = 0.65,
        theta_prune: float = 0.75,
        theta_gate: float = 0.3,
        K: int = 5,
        use_mlight: bool = True,
        enable_mlp_gate: bool = False,
        route_nonmatch: bool = True,
        stage1_feedback: str = "decisive",
        value_similarity_fn: Optional[Callable[[str, str], float]] = None,
        evidence_token_budget: int = 4000,
        decay_lambda: float = 0.01,
    ):
        self.hypergraph = hypergraph
        self.client = client

        self.theta_hi = theta_hi
        self.theta_prune = theta_prune
        self.theta_gate = theta_gate
        self.K = K
        self.use_mlight = use_mlight
        self.enable_mlp_gate = enable_mlp_gate
        self.route_nonmatch = route_nonmatch
        # How Stage-1 routed trajectories feed Eq. 8. "decisive" records s_R = 1
        # for the rule that produced the decision (no LLM was queried, and the
        # rule was by construction singularly decisive). "none" leaves the
        # numerator untouched, which penalises rules used by Stage 1.
        self.stage1_feedback = stage1_feedback
        self.value_similarity_fn = value_similarity_fn or self._token_overlap
        self.evidence_token_budget = evidence_token_budget
        self.decay_lambda = decay_lambda      # Eq. 9 freshness decay

        # Components injected by the plugin (paper Section III-C).
        self.mlight = None
        self.mheavy = None
        self.mlp_gate = None
        self.model = "gpt-3.5-turbo"      # overwritten by the plugin

        # Statistics
        self.stage1_total = 0
        self.stage1_hits = 0
        self.stage1_match_hits = 0
        self.stage1_nonmatch_hits = 0
        self.stage1_gate_hits = 0
        self.stage2_total = 0
        self.stage2_hits = 0
        self.trajectory_total = 0

    # ------------------------------------------------------------------ atoms
    @staticmethod
    def _token_overlap(v1: str, v2: str) -> float:
        t1, t2 = set(str(v1).lower().split()), set(str(v2).lower().split())
        if not t1 or not t2:
            return 0.0
        return len(t1 & t2) / len(t1 | t2)

    @staticmethod
    def _lookup(ctx: Dict, attr: str) -> str:
        """Schema-tolerant attribute lookup.

        Datasets differ in attribute naming conventions (e.g. ``artist_name``
        vs ``artist``), so a direct hit is tried first, then a case-insensitive
        hit, then token-level key containment. Missing attributes return "".
        """
        if not ctx or not attr:
            return ""
        if attr in ctx and isinstance(ctx[attr], str):
            return ctx[attr]

        lower_map = {}
        for k, v in ctx.items():
            if isinstance(v, str):
                lower_map.setdefault(k.lower(), v)
        key = attr.lower()
        if key in lower_map:
            return lower_map[key]

        a_tokens = set(key.replace("_", " ").split())
        for k, v in lower_map.items():
            k_tokens = set(k.replace("_", " ").split())
            if a_tokens & k_tokens:
                return v
        for k, v in lower_map.items():
            if len(key) >= 3 and (key in k or k in key):
                return v
        return ""

    def _typed_lookup(self, atom: ConditionAtom, es_ctx: Dict,
                      et_ctx: Dict) -> Tuple[str, str]:
        """Resolve a typed variable ``$NAME`` to the first attribute that yields values."""
        if not atom.attr.startswith("$"):
            return self._lookup(es_ctx, atom.attr), self._lookup(et_ctx, atom.attr)

        skip = {"entity_name", "name", "id", "entity_id"}
        for key in es_ctx:
            if not isinstance(es_ctx.get(key), str) or key.startswith("neighbors_"):
                continue
            if key.lower() in skip:
                continue
            v1, v2 = self._lookup(es_ctx, key), self._lookup(et_ctx, key)
            if v1 and v2:
                return v1, v2
        return "", ""

    def _verify_atom(self, atom: ConditionAtom, es_ctx: Dict, et_ctx: Dict) -> bool:
        """Evaluate one condition atom against an entity pair."""
        t = atom.atom_type

        if t == "SameValue":
            v1, v2 = self._typed_lookup(atom, es_ctx, et_ctx)
            if not v1 or not v2:
                return False
            a, b = v1.lower().strip(), v2.lower().strip()
            if not a or not b:
                return False
            if a == b or a in b or b in a:
                return True
            return self._token_overlap(v1, v2) >= 0.5

        if t == "DifferValue":
            v1, v2 = self._typed_lookup(atom, es_ctx, et_ctx)
            if not v1 or not v2:
                return False
            a, b = v1.lower().strip(), v2.lower().strip()
            if a == b:
                return False
            return self._token_overlap(v1, v2) < 0.4

        if t in ("ShareNeighbor", "DifferNeighbor"):
            n1 = es_ctx.get(f"neighbors_{atom.attr}", set()) or set()
            n2 = et_ctx.get(f"neighbors_{atom.attr}", set()) or set()
            if not isinstance(n1, (set, list, tuple)) or not isinstance(n2, (set, list, tuple)):
                return False
            shared = bool(set(n1) & set(n2))
            if t == "ShareNeighbor":
                return shared
            return bool(n1) and bool(n2) and not shared

        if t in ("SemanticEquiv", "SemanticConflict"):
            v1, v2 = self._typed_lookup(atom, es_ctx, et_ctx)
            if not v1 or not v2:
                return False
            sim = self.value_similarity_fn(v1, v2)
            thr = atom.threshold if atom.threshold > 0 else (
                0.5 if t == "SemanticEquiv" else 0.2)
            return sim >= thr if t == "SemanticEquiv" else sim < thr

        return False

    def verify_rule(self, rule: FOLRule, es_ctx: Dict, et_ctx: Dict) -> bool:
        """True iff every atom in the rule body holds (conjunction semantics)."""
        if not rule.atoms:
            return False
        return all(self._verify_atom(a, es_ctx, et_ctx) for a in rule.atoms)

    def trigger_rules(self, es_id: str, et_id: str,
                      es_ctx: Dict, et_ctx: Dict) -> Tuple[List[FOLRule], List[FOLRule]]:
        """Retrieve candidate rules and keep the triggered ones."""
        candidates = self.hypergraph.get_candidate_rules([es_id, et_id], [es_ctx, et_ctx])
        triggered = [r for r in candidates if self.verify_rule(r, es_ctx, et_ctx)]
        return triggered, candidates# ------------------------------------------------------------------ stage 1
    def stage1_symbolic_filtering(self, es_id: str, et_id: str,
                                  es_context: Dict, et_context: Dict
                                  ) -> Tuple[Optional[int], List[FOLRule], List[FOLRule]]:
        """Symbolic filtering and gated routing (Section III-E, Stage 1).

        Returns ``(decision, triggered_rules, candidate_rules)`` where
        ``decision`` is ``None`` when the pair survives to Stage 2.
        """
        self.stage1_total += 1

        triggered, candidates = self.trigger_rules(es_id, et_id, es_context, et_context)
        if not triggered:
            return None, [], candidates

        match_rules = [r for r in triggered if r.conclusion == 1]
        nonmatch_rules = [r for r in triggered if r.conclusion == 0]

        # Direct matching: unanimous positive verdict, confidence >= theta_hi
        if match_rules and not nonmatch_rules:
            if max(r.conf for r in match_rules) >= self.theta_hi:
                self.stage1_hits += 1
                self.stage1_match_hits += 1
                return 1, triggered, candidates

        # Direct non-matching: unanimous negative verdict, confidence >= theta_prune
        if self.route_nonmatch and nonmatch_rules and not match_rules:
            if max(r.conf for r in nonmatch_rules) >= self.theta_prune:
                self.stage1_hits += 1
                self.stage1_nonmatch_hits += 1
                return 0, triggered, candidates

        # Survival -> MLP gate (Section III-E, "MLP gating")
        if self.enable_mlp_gate and self.mlp_gate is not None and self.mlp_gate.is_trained:
            try:
                feats = self.mlp_gate.extract_features(
                    es_context, et_context, triggered,
                    es_embedding=self.hypergraph.entity_embeddings.get(f"e_{es_id}"),
                    et_embedding=self.hypergraph.entity_embeddings.get(f"e_{et_id}"),
                    hypergraph=self.hypergraph)
                if not self.mlp_gate.should_invoke_llm(feats):
                    self.stage1_hits += 1
                    self.stage1_gate_hits += 1
                    return 0, triggered, candidates
            except Exception as exc:      # gate failures must never break inference
                print(f"[EvoRM] MLP gate skipped: {exc}")

        return None, triggered, candidates

    # ------------------------------------------------------------------ evidence
    def construct_evidence_subgraph(self, es_id: str, et_id: str,
                                    es_context: Dict, et_context: Dict,
                                    candidate_rules: Optional[List[FOLRule]] = None) -> Dict:
        """E_sub: top-K highest-weight hyperedges plus their node sets (Section III-E)."""
        edges = self.hypergraph.get_candidate_hyperedges(
            [es_id, et_id], [es_context, et_context])

        joint = [he for he in edges if f"e_{es_id}" in he.node_set and f"e_{et_id}" in he.node_set]
        joint.sort(key=lambda h: h.weight, reverse=True)

        selected = list(joint[: self.K])
        if len(selected) < self.K:
            rest = [he for he in edges if he not in selected]
            rest.sort(key=lambda h: h.weight, reverse=True)
            selected.extend(rest[: self.K - len(selected)])

        return {
            "hyperedges": selected,
            "nodes": sorted({n for he in selected for n in he.node_set}),
            "rules": candidate_rules or [],
        }

    def build_stage2_prompt(self, es_id: str, et_id: str,
                            es_context: Dict, et_context: Dict,
                            evidence: Dict,
                            triggered_rules: List[FOLRule],
                            base_prompt: str) -> str:
        """Serialise E_sub and R_trig into the decision prompt T_dec."""
        rule_section = ""
        if triggered_rules:
            rule_section = "\n\n**Triggered Rules (from historical matching experience):**\n"
            for idx, rule in enumerate(triggered_rules, 1):
                body = " ∧ ".join(f"{a.atom_type}({a.attr})" for a in rule.atoms)
                verdict = "MATCH" if rule.conclusion == 1 else "NON-MATCH"
                rule_section += (
                    f"  Rule {idx}: {body} → {verdict}\n"
                    f"    Confidence: {rule.conf:.3f} | Triggered: {rule.trigger_count} times\n"
                )

        evidence_section = ""
        if evidence.get("hyperedges"):
            evidence_section = "\n\n**Evidence from Similar Historical Cases:**\n"
            for he in evidence["hyperedges"][:3]:
                evidence_section += (
                    f"  - Cluster (weight={he.weight:.3f}): "
                    f"{len(he.entity_pairs)} similar entity pairs stored\n"
                )

        feedback = """

**IMPORTANT: Rule Contribution Feedback**
After making your decision, you MUST also provide a JSON object with contribution scores (0.0 to 1.0) for each triggered rule that influenced your decision. Format:
```json
{
  "decision": "MATCH" or "NON-MATCH",
  "rule_feedback": {
    "rule_1": 0.8,
    "rule_2": 0.3
  },
  "reasoning": "[DECISIVE] ... [SUPPORTING] ..."
}
```
Score meaning: 1.0 = DECISIVE, 0.1-0.9 = SUPPORTING, 0.0 = IRRELEVANT.
You MUST respond with valid JSON only."""

        return base_prompt + rule_section + evidence_section + feedback

    def parse_llm_feedback(self, response_text: str
                           ) -> Tuple[Optional[int], Dict[str, float], str]:
        """Parse ``{"decision": ..., "rule_feedback": {...}, "reasoning": ...}``."""
        try:
            block = re.search(r"```json\s*(.*?)\s*```", response_text, re.DOTALL)
            raw = block.group(1) if block else None
            if raw is None:
                bare = re.search(r'\{.*"decision".*\}', response_text, re.DOTALL)
                raw = bare.group(0) if bare else response_text
            data = json.loads(raw)

            decision_str = str(data.get("decision", "")).upper()
            if "NON" in decision_str or "NO MATCH" in decision_str:
                decision = 0
            elif "MATCH" in decision_str:
                decision = 1
            else:
                decision = None

            feedback = {str(k): float(v) for k, v in (data.get("rule_feedback") or {}).items()
                        if isinstance(v, (int, float))}
            return decision, feedback, str(data.get("reasoning", ""))
        except Exception:
            pass

        low = response_text.lower()
        if "non-match" in low or "no match" in low:
            return 0, {}, response_text
        if "match" in low:
            return 1, {}, response_text
        return None, {}, response_text

    # ------------------------------------------------------------------ stage 2
    def stage2_elicit_rationale(self, es_id: str, et_id: str,
                                es_context: Dict, et_context: Dict,
                                triggered_rules: List[FOLRule],
                                decision: Optional[int] = None):
        """Second half of the Stage-2 pair: M_light externalises the evidence
        behind ``decision`` (Section III-C: r = M_light(T_enc(..., y_hat)))."""
        if self.mlight is not None:
            return self.mlight.elicit(
                es_context, et_context, triggered_rules, decision=decision)
        from .mlight import RationaleResult
        return RationaleResult(
            rationale="[DECISIVE] direct attribute comparison\n"
                      "[SUPPORTING] Mlight unavailable, template fallback",
            token_usage=0,
        )

    def stage2_llm_judgment(self, es_id: str, et_id: str,
                            es_context: Dict, et_context: Dict,
                            triggered_rules: List[FOLRule],
                            candidate_rules: List[FOLRule],
                            base_prompt: str = "") -> Dict:
        """Run the Stage-2 LLM pair in the order fixed by Section III-C:

            y_hat = M_heavy(T_dec(e_s, e_t, E_sub, R_trig))
            r     = M_light(T_enc(e_s, e_t, ctx(e_s, e_t), y_hat))

        The decision comes first; the rationale explains it and is what
        RuleEncoding later converts into a rule.
        """
        self.stage2_total += 1

        # ------------------------------------------------ Mheavy -> Mlight
        if self.use_mlight and self.mlight is not None and self.mheavy is not None:
            decision_result = self.mheavy.decide(
                es_context, et_context, None, triggered_rules,
                base_prompt=base_prompt)
            tokens = decision_result.get("token_usage", 0)

            if decision_result.get("decision") is not None:
                self.stage2_hits += 1

            decision_value = decision_result.get("decision")
            if triggered_rules or decision_value is not None:
                rationale_result = self.stage2_elicit_rationale(
                    es_id, et_id, es_context, et_context, triggered_rules,
                    decision=decision_value)
                rationale_text = rationale_result.rationale
                tokens += rationale_result.token_usage
            else:
                rationale_text = ("[DECISIVE] Direct attribute comparison\n"
                                  "[SUPPORTING] No historical rules available")

            rationale = rationale_text
            if not triggered_rules:
                summary = decision_result.get("summary", "")
                if summary and summary != "no client available":
                    rationale = f"[DECISIVE] {summary}\n[SUPPORTING] Direct LLM reasoning"

            feedback = self._map_feedback(
                triggered_rules, decision_result.get("rule_feedback", {}))
            return {
                "decision": decision_result.get("decision"),
                "rule_feedback": feedback,
                "rationale": rationale,
                "confidence": decision_result.get("confidence", 0.5),
                "token_usage": tokens,
                "raw_response": decision_result.get("raw_response", ""),
                "method": "mheavy+mlight" if triggered_rules else "mheavy-only",
            }

        # ------------------------------------------------ legacy single call
        evidence = self.construct_evidence_subgraph(
            es_id, et_id, es_context, et_context, candidate_rules)
        prompt = self.build_stage2_prompt(
            es_id, et_id, es_context, et_context,
            evidence, triggered_rules, base_prompt)

        if self.client is None:
            return {"decision": None, "rule_feedback": {}, "rationale": "",
                    "token_usage": 0, "raw_response": "", "method": "no-client"}

        try:
            response = self.client.chat.completions.create(
                model=getattr(self, "model", "gpt-3.5-turbo"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            text = response.choices[0].message.content.strip()
            decision, feedback, rationale = self.parse_llm_feedback(text)
            if decision is not None:
                self.stage2_hits += 1
            return {
                "decision": decision,
                "rule_feedback": self._map_feedback(triggered_rules, feedback),
                "rationale": rationale,
                "token_usage": response.usage.total_tokens,
                "raw_response": text,
                "method": "single-call",
            }
        except Exception as exc:
            return {"decision": None, "rule_feedback": {}, "rationale": "",
                    "token_usage": 0, "raw_response": "",
                    "error": str(exc), "method": "error"}

    @staticmethod
    def _map_feedback(triggered_rules: List[FOLRule],
                      raw_feedback: Dict) -> Dict[str, float]:
        """Map ``rule_1..rule_n`` / rule ids returned by the LLM onto rule ids."""
        mapped: Dict[str, float] = {}
        for key, value in (raw_feedback or {}).items():
            key = str(key).strip()
            if key in {r.rule_id for r in triggered_rules}:
                mapped[key] = float(value)
                continue
            match = re.search(r"(\d+)$", key)
            if match:
                idx = int(match.group(1)) - 1
                if 0 <= idx < len(triggered_rules):
                    mapped[triggered_rules[idx].rule_id] = float(value)
        return mapped

    # ------------------------------------------------------------------ Eq. 8
    def record_trajectory(self, triggered_rules: List[FOLRule],
                          rule_feedback: Optional[Dict[str, float]] = None,
                          decisive_rule: Optional[FOLRule] = None):
        """Update rule statistics for one matching trajectory (Section III-F).

        ``trigger_count`` counts every trajectory in which the rule was
        triggered (the denominator of Eq. 8); ``sR_sum`` accumulates the
        contribution scores of the trajectories where it was used (numerator).
        """
        import time as _time

        self.trajectory_total += 1
        feedback = rule_feedback or {}

        for rule in triggered_rules:
            rule.trigger_count += 1

            s = 0.0
            if decisive_rule is not None and self.stage1_feedback == "decisive":
                if rule.rule_id == decisive_rule.rule_id:
                    s = 1.0
            else:
                s = float(feedback.get(rule.rule_id, 0.0))

            if s > 0:
                rule.sR_sum += s
                rule.used_count += 1
                rule.last_used_time = _time.time()

            rule.update_confidence()
            rule.update_freshness(self.decay_lambda)

    # ------------------------------------------------------------------ stats
    def get_stats(self) -> Dict:
        stats = {
            "stage1_total": self.stage1_total,
            "stage1_hits": self.stage1_hits,
            "stage1_rate": self.stage1_hits / max(1, self.stage1_total),
            "stage1_match_hits": self.stage1_match_hits,
            "stage1_nonmatch_hits": self.stage1_nonmatch_hits,
            "stage1_gate_hits": self.stage1_gate_hits,
            "stage2_total": self.stage2_total,
            "stage2_hits": self.stage2_hits,
            "trajectory_total": self.trajectory_total,
            "use_mlight": self.use_mlight,
        }
        if self.mlight is not None:
            stats["mlight"] = self.mlight.get_stats()
        if self.mheavy is not None:
            stats["mheavy"] = self.mheavy.get_stats()
        if self.mlp_gate is not None:
            stats["mlp_gate"] = self.mlp_gate.get_stats()
        return stats