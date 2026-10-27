#!/usr/bin/env python3
"""EvoRM plugin -- assembles the four components of Section III."""

import time
from typing import Callable, Dict, Optional

from .config import EvoRMConfig
from .controller import TwoStageInferenceController
from .entity_embedding import EntityEmbedder
from .hypergraph import HypergraphStorage
from .maintenance import RuleMaintenance
from .mlight import MheavyDecisionMaker, MlightRationaleElicitor
from .mlp_gate import MLPGate
from .rule_encoding import RuleEncoding
from .types import FOLRule


class EvoRMPlugin:
    """End-to-end EvoRM: rule encoding, hypergraph memory, controller, maintenance."""

    def __init__(self, config: Optional[EvoRMConfig] = None,
                 client=None, mlight_client=None, verbose: bool = False,
                 embedding_fn: Optional[Callable[[Dict, str], object]] = None):
        self.config = config or EvoRMConfig()
        self.verbose = verbose
        self.client = client
        self.mlight_client = mlight_client or client

        cfg = self.config
        self.ablation_mode = cfg.ablation_mode

        # --- component (1) Rule Encoding, Section III-C
        self.rule_encoding = RuleEncoding(
            client=self.mlight_client, model=cfg.mlight_model,
            temperature=cfg.temperature)

        # --- component (2) Hypergraph Storage, Section III-D
        # `embedding_fn(context, entity_id) -> vector` lets you plug a real
        # encoder (Sentence-BERT, an LLM embedding endpoint, ...). The default
        # is the deterministic hashing embedder shipped with the repository.
        self.embedding_fn = embedding_fn
        self.embedder = EntityEmbedder(demb=cfg.demb, n_features=cfg.n_features)
        self.hypergraph = HypergraphStorage(
            alpha=cfg.alpha, beta=cfg.beta,
            merge_threshold=cfg.merge_threshold,
            entity_embedder=self.embedder,
            enable_attr_fallback=cfg.enable_attr_fallback,
            flat_mode=(cfg.ablation_mode == "no_hypergraph"))

        # --- component (3) Two-Stage Inference Controller, Section III-E
        self.controller = TwoStageInferenceController(
            hypergraph=self.hypergraph,
            client=client,
            theta_hi=cfg.theta_hi,
            theta_prune=cfg.theta_prune,
            theta_gate=cfg.theta_gate,
            K=cfg.K,
            use_mlight=cfg.use_mlight,
            enable_mlp_gate=cfg.enable_mlp_gate and cfg.ablation_mode != "no_stage1",
            route_nonmatch=cfg.route_nonmatch,
            stage1_feedback=cfg.stage1_feedback,
            decay_lambda=cfg.decay_lambda,
        )

        # Mlight / Mheavy (Section III-C)
        self.mlight = MlightRationaleElicitor(
            client=self.mlight_client, model=cfg.mlight_model,
            temperature=cfg.temperature)
        self.mheavy = MheavyDecisionMaker(
            client=client, model=cfg.model, temperature=cfg.temperature)
        self.controller.mlight = self.mlight
        self.controller.mheavy = self.mheavy
        self.controller.model = cfg.model

        # MLP gate g_phi (Section III-E)
        self.mlp_gate: Optional[MLPGate] = None
        if cfg.enable_mlp_gate and cfg.ablation_mode not in ("no_stage1",):
            try:
                self.mlp_gate = MLPGate(
                    theta_gate=cfg.theta_gate,
                    n_warmup=cfg.mlp_n_warmup,
                    demb=cfg.demb)
                self.controller.mlp_gate = self.mlp_gate
            except ImportError as exc:
                if verbose:
                    print(f"[EvoRM] MLP gate disabled: {exc}")

        # --- component (4) Rule Maintenance, Section III-F
        self.maintenance = RuleMaintenance(
            hypergraph=self.hypergraph,
            decay_lambda=cfg.decay_lambda,
            freshness_threshold=cfg.freshness_threshold,
            confidence_threshold=cfg.confidence_threshold,
            eval_triggers=cfg.eval_triggers,
            merge_similarity=cfg.merge_similarity,
            max_rules=cfg.max_rules,
            evict_percentile=cfg.evict_percentile,
            optimization_interval=cfg.optimization_interval,
            warm_start_conf=cfg.warm_start_conf,
            meta_rule_fn=self._meta_rule_via_mlight,
            verbose=verbose,
        )

        # Bookkeeping
        self.query_count = 0
        self.total_tokens = 0

    # ------------------------------------------------------------------ memory
    def _node_set(self, es_id: str, et_id: str, rule: FOLRule) -> set:
        """V_R = V_E ∪ V_A ∪ V_V for this rule, plus its graph-dependent nodes.

        V_E holds the two entity nodes, V_A one node per attribute mentioned by
        the rule, V_V one node per concrete value the rule asserts is shared.
        """
        nodes = {f"e_{es_id}", f"e_{et_id}"}
        for atom in rule.atoms:
            attr = atom.attr.lower()
            nodes.add(f"a_{attr}")
            if atom.atom_type in ("ShareNeighbor", "DifferNeighbor"):
                nodes.add(f"n_{attr}")
            elif atom.atom_type in ("SameValue", "SemanticEquiv") and atom.value1:
                nodes.add(f"v_{attr}={str(atom.value1).strip().lower()}")
        return nodes

    def _remember_entities(self, es_id: str, et_id: str,
                           es_ctx: Dict, et_ctx: Dict):
        def encode(ctx, eid):
            if self.embedding_fn is not None:
                return self.embedding_fn(ctx, eid)
            return self.embedder.embed_entity(ctx, eid)

        self.hypergraph.store_entity_embedding(f"e_{es_id}", encode(es_ctx, str(es_id)))
        self.hypergraph.store_entity_embedding(f"e_{et_id}", encode(et_ctx, str(et_id)))

    # ------------------------------------------------------------------ meta-rule
    def _meta_rule_via_mlight(self, cluster) -> Optional[FOLRule]:
        """Eq. 10 meta-rule: Mlight generalises the cluster into one abstract rule."""
        if self.mlight_client is None:
            return None
        return self.rule_encoding.abstract_meta_rule(cluster, client=self.mlight_client)

    # ------------------------------------------------------------------ inference
    def process(self, es_id: str, et_id: str,
                es_context: Dict, et_context: Dict,
                base_prompt: str = "",
                base_matcher: Optional[Callable] = None,
                gold: Optional[int] = None,
                learn: bool = True) -> Dict:
        """Run one matching trajectory through the full EvoRM pipeline.

        ``base_matcher`` is the host-model port: a callable (or object exposing
        ``.match``) that receives a request dict and returns a response dict with
        at least ``decision``; optionally ``rule_feedback``, ``rationale``,
        ``token_usage``. When omitted, the built-in Mlight+Mheavy pair is used.
        """
        started = time.time()
        cfg = self.config

        self._remember_entities(es_id, et_id, es_context, et_context)

        # ---------------- Stage 1: symbolic filtering and gated routing
        if cfg.ablation_mode == "no_stage1":
            triggered, candidates = self.controller.trigger_rules(
                es_id, et_id, es_context, et_context)
            decision, decisive_rule = None, None
        else:
            decision, triggered, candidates = self.controller.stage1_symbolic_filtering(
                es_id, et_id, es_context, et_context)
            decisive_rule = None
            if decision is not None:
                pool = [r for r in triggered if r.conclusion == decision]
                if pool:
                    decisive_rule = max(pool, key=lambda r: r.conf)

        result = {
            "decision": decision,
            "stage": "stage1" if decision is not None else "stage2",
            "confidence": 0.0,
            "rationale": "",
            "tokens": 0,
            "triggered_rules": [r.rule_id for r in triggered],
            "new_rule": None,
            "latency_s": 0.0,
        }

        if decision is not None:
            result["confidence"] = decisive_rule.conf if decisive_rule else 0.0
            result["rationale"] = "stage-1 symbolic routing"
            result["latency_s"] = time.time() - started
            self.query_count += 1

            if learn:
                self.controller.record_trajectory(
                    triggered, rule_feedback=None, decisive_rule=decisive_rule)
                self._maybe_optimize()
            return result

        # ---------------- Stage 2: LLM judgment
        if base_matcher is not None:
            stage2 = self._call_base_matcher(
                base_matcher, es_id, et_id, es_context, et_context,
                triggered, candidates, base_prompt, gold)
        else:
            stage2 = self.controller.stage2_llm_judgment(
                es_id, et_id, es_context, et_context,
                triggered, candidates, base_prompt)

        result["decision"] = stage2.get("decision")
        result["confidence"] = stage2.get("confidence", 0.5)
        result["rationale"] = stage2.get("rationale", "")
        result["tokens"] = int(stage2.get("token_usage", 0))
        result["method"] = stage2.get("method", "")
        self.total_tokens += result["tokens"]
        self.query_count += 1

        # ---------------- Rule Encoding + memory update
        if learn and result["decision"] is not None and result["rationale"]:
            new_rule = self.rule_encoding.create_rule(
                rationale=result["rationale"],
                conclusion=int(result["decision"]),
                es_context=es_context,
                et_context=et_context,
                used_rules=[(r, r.conf) for r in triggered],
            )
            if new_rule is not None:
                nodes = self._node_set(es_id, et_id, new_rule)
                self.hypergraph.add_trajectory((es_id, et_id), nodes, new_rule)
                result["new_rule"] = new_rule.rule_id

        if learn:
            feedback = stage2.get("rule_feedback") or {}
            if not feedback and result["decision"] is not None:
                # The external matcher did not score the rules it saw; rules
                # that agree with its verdict are the ones that were decisive.
                feedback = {r.rule_id: 1.0 for r in triggered
                            if r.conclusion == int(result["decision"])}
            self.controller.record_trajectory(triggered, rule_feedback=feedback)
            self._maybe_optimize()

            if self.mlp_gate is not None and result["decision"] is not None:
                self.mlp_gate.collect_sample(
                    self._gate_features(es_id, et_id, es_context, et_context, triggered),
                    float(result["decision"]))
                if self.mlp_gate.is_ready_to_train():
                    self.mlp_gate.fit_model(
                        epochs=cfg.mlp_epochs, batch_size=cfg.mlp_batch_size)

        result["latency_s"] = time.time() - started
        return result

    def _gate_features(self, es_id, et_id, es_ctx, et_ctx, triggered):
        return self.mlp_gate.extract_features(
            es_ctx, et_ctx, triggered,
            es_embedding=self.hypergraph.entity_embeddings.get(f"e_{es_id}"),
            et_embedding=self.hypergraph.entity_embeddings.get(f"e_{et_id}"),
            hypergraph=self.hypergraph)

    def _call_base_matcher(self, matcher, es_id, et_id, es_ctx, et_ctx,
                           triggered, candidates, base_prompt, gold) -> Dict:
        """Adapt an external matcher to the internal Stage-2 response format."""
        evidence = self.controller.construct_evidence_subgraph(
            es_id, et_id, es_ctx, et_ctx, candidates)
        request = {
            "es_id": es_id, "et_id": et_id,
            "es_context": es_ctx, "et_context": et_ctx,
            "triggered_rules": [r.to_dict() for r in triggered],
            "evidence": [he.hyperedge_id for he in evidence["hyperedges"]],
            "prompt": base_prompt,
            "gold": gold,
        }
        raw = matcher.match(request) if hasattr(matcher, "match") else matcher(request)
        raw = raw or {}
        return {
            "decision": raw.get("decision"),
            "rule_feedback": raw.get("rule_feedback", {}),
            "rationale": raw.get("rationale", ""),
            "confidence": raw.get("confidence", 0.5),
            "token_usage": raw.get("token_usage", raw.get("tokens", 0)),
            "method": raw.get("method", "external"),
        }

    # ------------------------------------------------------------------ maintenance
    def _maybe_optimize(self):
        if self.config.ablation_mode == "no_maintenance":
            return
        if self.maintenance.should_optimize():
            self.maintenance.optimize()
            self.hypergraph.refresh_weights()

    # ------------------------------------------------------------------ persistence
    def save(self, directory: str):
        import os
        os.makedirs(directory, exist_ok=True)
        self.config.save(os.path.join(directory, "config.json"))
        import json
        with open(os.path.join(directory, "hypergraph.json"), "w", encoding="utf-8") as fh:
            json.dump(self.hypergraph.to_dict(), fh, indent=2)
        if self.mlp_gate is not None:
            self.mlp_gate.save(os.path.join(directory, "mlp_gate.pt"))

    def load(self, directory: str):
        import os
        path = os.path.join(directory, "hypergraph.json")
        if not os.path.exists(path):
            return False
        import json
        with open(path, "r", encoding="utf-8") as fh:
            state = HypergraphStorage.from_dict(json.load(fh), entity_embedder=self.embedder,
                                                enable_attr_fallback=self.config.enable_attr_fallback)
        self.hypergraph = state
        self.controller.hypergraph = state
        self.maintenance.hypergraph = state
        if self.mlp_gate is not None:
            self.mlp_gate.load(os.path.join(directory, "mlp_gate.pt"))
        return True

    # ------------------------------------------------------------------ stats
    def stats(self) -> Dict:
        return {
            "queries": self.query_count,
            "total_tokens": self.total_tokens,
            "avg_tokens": self.total_tokens / max(1, self.query_count),
            "hypergraph": self.hypergraph.stats(),
            "controller": self.controller.get_stats(),
            "maintenance": self.maintenance.get_stats(),
        }