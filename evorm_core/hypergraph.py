#!/usr/bin/env python3
"""
Hypergraph Storage -- paper Section III-D.

H_t = (V, R, E_hyper), where
    V       = V_E u V_A u V_V   entity / attribute-name / attribute-value nodes
    R       global rule set
    E_hyper set of hyperedges H_k = (M_k, V_k, R_k, w_k)

Highlights
----------
* w_k follows Eq. 6: alpha * max_{R in R_k} Conf(R) + beta * mean_{(ei,ej) in M_k} cos(ei, ej)
* Hyperedge creation/update uses the maximal Jaccard node overlap (Eq. 7) with
  threshold eta_h.
* Retrieval is O(1) through an inverted index (entity -> hyperedge ids).
"""

from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Set, Tuple

from .types import FOLRule, Hyperedge


class HypergraphStorage:
    """Weighted hypergraph organising rules and entity contexts."""

    def __init__(self, alpha: float = 0.6, beta: float = 0.4,
                 merge_threshold: float = 0.5,
                 entity_embedder=None,
                 enable_attr_fallback: bool = True,
                 flat_mode: bool = False):
        self.hyperedges: Dict[str, Hyperedge] = {}
        self.rules: Dict[str, FOLRule] = {}
        # Inverted index: node id -> set of hyperedge ids
        self.inverted_index: Dict[str, Set[str]] = defaultdict(set)

        self.alpha = alpha
        self.beta = beta
        self.merge_threshold = merge_threshold      # eta_h
        self.edge_counter = 0

        self.entity_embedder = entity_embedder
        self.entity_embeddings: Dict[str, "object"] = {}
        self.flat_mode = flat_mode                  # ablation "w/o U"
        # Datasets where every entity id appears in exactly one pair (ER) cannot
        # exploit pure entity-based retrieval. This flag enables an auxiliary
        # schema-level lookup over attribute names so that learned rules remain
        # reusable; it is reported separately in the paper's appendix.
        self.enable_attr_fallback = enable_attr_fallback

    # ------------------------------------------------------------------ indexing
    def _next_edge_id(self) -> str:
        self.edge_counter += 1
        return f"HE_{self.edge_counter:06d}"

    def _index(self, edge_id: str, node_set: Iterable[str], rule: FOLRule):
        for node in node_set:
            self.inverted_index[node].add(edge_id)
        for atom in rule.atoms:
            self.inverted_index[f"__attr__::{atom.attr.lower()}"].add(edge_id)

    def register_rule(self, rule: FOLRule):
        """Insert a rule into the global rule set (idempotent)."""
        self.rules.setdefault(rule.rule_id, rule)

    def store_entity_embedding(self, entity_id: str, embedding):
        self.entity_embeddings[entity_id] = embedding

    def get_entity_embeddings(self) -> Dict:
        return self.entity_embeddings

    # ------------------------------------------------------------------ memory
    def add_trajectory(
        self,
        entity_pair: Tuple[int, int],
        node_set: Set[str],
        rule: FOLRule,
    ) -> Hyperedge:
        """Insert (or absorb) one trajectory into the hypergraph.

        Implements Section III-D "Hyperedge Creation/Update":
            k* = argmax_k |V_R n V_k| / |V_R u V_k|
            if overlap >= eta_h: update H_k* ; else create H_new
        """
        self.register_rule(rule)

        if self.flat_mode:
            return Hyperedge(
                hyperedge_id="flat", entity_pairs={entity_pair},
                node_set=set(node_set), rules=[rule],
                weight=max(r.conf for r in [rule]),
            )

        best_id, best_overlap = None, 0.0
        for he_id, he in self.hyperedges.items():
            if not he.node_set:
                continue
            union = len(node_set | he.node_set)
            if union == 0:
                continue
            jaccard = len(node_set & he.node_set) / union
            if jaccard > best_overlap:
                best_overlap, best_id = jaccard, he_id

        if best_id is not None and best_overlap >= self.merge_threshold:
            he = self.hyperedges[best_id]
            he.entity_pairs.add(entity_pair)
            he.node_set.update(node_set)
            if all(r.rule_id != rule.rule_id for r in he.rules):
                he.rules.append(rule)
            he.recompute_weight(
                self.alpha, self.beta, self.entity_embeddings,
                similarity_fn=self._cosine)
            self._index(best_id, node_set, rule)
            return he

        he = Hyperedge(
            hyperedge_id=self._next_edge_id(),
            entity_pairs={entity_pair},
            node_set=set(node_set),
            rules=[rule],
        )
        he.recompute_weight(
            self.alpha, self.beta, self.entity_embeddings,
            similarity_fn=self._cosine)
        self.hyperedges[he.hyperedge_id] = he
        self._index(he.hyperedge_id, node_set, rule)
        return he

    @staticmethod
    def _cosine(v1, v2) -> float:
        try:
            import numpy as np
            a, b = np.asarray(v1, dtype=float), np.asarray(v2, dtype=float)
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na == 0 or nb == 0:
                return 0.0
            return float(np.dot(a, b) / (na * nb))
        except Exception:
            return 0.0

    def refresh_weights(self):
        """Recompute every hyperedge weight (called after rule confidence updates)."""
        for he in self.hyperedges.values():
            he.recompute_weight(
                self.alpha, self.beta, self.entity_embeddings,
                similarity_fn=self._cosine)

    # ------------------------------------------------------------------ retrieval
    def get_candidate_hyperedges(self, entity_ids: List[str],
                                 entity_contexts: Optional[List[Dict]] = None) -> List[Hyperedge]:
        """E_cand = {H_k | e_s in V_k or e_t in V_k} in O(1) per lookup."""
        candidate_ids: Set[str] = set()
        for eid in entity_ids:
            candidate_ids.update(self.inverted_index.get(f"e_{eid}", set()))

        if self.enable_attr_fallback and entity_contexts:
            attr_keys: Set[str] = set()
            for ctx in entity_contexts:
                if not ctx:
                    continue
                for key in ctx:
                    if key.startswith("neighbors_") or key.startswith("__"):
                        continue
                    attr_keys.add(key.lower())
            for key in attr_keys:
                candidate_ids.update(self.inverted_index.get(f"__attr__::{key}", set()))

        # Iterate the hyperedge store rather than the id set: set iteration order
        # depends on string hash randomisation, and a stable order keeps rule
        # retrieval, tie-breaking and learned confidences reproducible run to run.
        return [he for he_id, he in self.hyperedges.items() if he_id in candidate_ids]

    def get_candidate_rules(self, entity_ids: List[str],
                            entity_contexts: Optional[List[Dict]] = None) -> List[FOLRule]:
        """R_cand = union_{H_k in E_cand} R_k."""
        if self.flat_mode:
            return list(self.rules.values())

        seen: Dict[str, FOLRule] = {}
        for he in self.get_candidate_hyperedges(entity_ids, entity_contexts):
            for rule in he.rules:
                seen.setdefault(rule.rule_id, rule)
        return list(seen.values())

    def get_rule(self, rule_id: str) -> Optional[FOLRule]:
        return self.rules.get(rule_id)

    def remove_rule(self, rule_id: str):
        """Delete a rule from the global set and from every hyperedge."""
        self.rules.pop(rule_id, None)
        for he in self.hyperedges.values():
            he.rules = [r for r in he.rules if r.rule_id != rule_id]

    # ------------------------------------------------------------------ stats
    def stats(self) -> Dict:
        n_edges = len(self.hyperedges)
        return {
            "num_hyperedges": n_edges,
            "num_rules": len(self.rules),
            "num_index_keys": len(self.inverted_index),
            "avg_rules_per_edge": (
                sum(len(he.rules) for he in self.hyperedges.values()) / max(1, n_edges)),
            "avg_weight": (
                sum(he.weight for he in self.hyperedges.values()) / max(1, n_edges)),
        }

    # ------------------------------------------------------------------ persistence
    def to_dict(self) -> Dict:
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "merge_threshold": self.merge_threshold,
            "flat_mode": self.flat_mode,
            "rules": [r.to_dict() for r in self.rules.values()],
            "hyperedges": [
                {
                    "hyperedge_id": he.hyperedge_id,
                    "entity_pairs": [list(p) for p in he.entity_pairs],
                    "node_set": sorted(he.node_set),
                    "rule_ids": [r.rule_id for r in he.rules],
                    "weight": he.weight,
                }
                for he in self.hyperedges.values()
            ],
        }

    @classmethod
    def from_dict(cls, d: Dict, **kwargs) -> "HypergraphStorage":
        hg = cls(
            alpha=d.get("alpha", 0.6),
            beta=d.get("beta", 0.4),
            merge_threshold=d.get("merge_threshold", 0.5),
            flat_mode=d.get("flat_mode", False),
            **kwargs,
        )
        for rd in d.get("rules", []):
            rule = FOLRule.from_dict(rd)
            hg.rules[rule.rule_id] = rule
        for hed in d.get("hyperedges", []):
            he = Hyperedge(
                hyperedge_id=hed["hyperedge_id"],
                entity_pairs={tuple(p) for p in hed.get("entity_pairs", [])},
                node_set=set(hed.get("node_set", [])),
                rules=[hg.rules[rid] for rid in hed.get("rule_ids", []) if rid in hg.rules],
                weight=float(hed.get("weight", 0.0)),
            )
            hg.hyperedges[he.hyperedge_id] = he
            for node in he.node_set:
                hg.inverted_index[node].add(he.hyperedge_id)
            for rule in he.rules:
                for atom in rule.atoms:
                    hg.inverted_index[f"__attr__::{atom.attr.lower()}"].add(he.hyperedge_id)
        hg.edge_counter = len(hg.hyperedges)
        return hg