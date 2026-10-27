#!/usr/bin/env python3
"""
Rule Maintenance -- paper Section III-F.

Two phases:

1. Dynamic state tracking (asynchronous)
       Conf(R)   Eq. 8   -- updated with every LLM contribution score s_R
       Fresh(R)  Eq. 9   -- exponential decay from the last positive s_R

2. Periodic set optimisation (four operations)
       pruning        : Fresh(R) < theta_f            -> purge
       flipping       : Conf(R) < theta_c with enough triggers
                        -> flip 1 -> 0, reset, re-evaluate after N_eval
       merging        : Jaccard(A(R_i), A(R_j)) >= eta_m
                        -> Mlight meta-rule, Conf Eq. 10
       forgetting     : |R| > |R|_max -> evict bottom gamma by Conf * Fresh
"""

import time
from itertools import combinations
from typing import Callable, Dict, List, Optional

from .types import ConditionAtom, FOLRule, new_id


class RuleMaintenance:
    """Confidence tracking plus periodic pruning / flipping / merging / forgetting."""

    def __init__(self,
                 hypergraph,
                 decay_lambda: float = 0.01,
                 freshness_threshold: float = 0.1,
                 confidence_threshold: float = 0.2,
                 eval_triggers: int = 10,
                 merge_similarity: float = 0.7,
                 max_rules: int = 10000,
                 evict_percentile: float = 0.1,
                 optimization_interval: float = 300.0,
                 warm_start_conf: float = 0.3,
                 meta_rule_fn: Optional[Callable[[List[FOLRule]], Optional[FOLRule]]] = None,
                 verbose: bool = False):
        self.hypergraph = hypergraph
        self.decay_lambda = decay_lambda
        self.theta_f = freshness_threshold
        self.theta_c = confidence_threshold
        self.N_eval = eval_triggers
        self.eta_m = merge_similarity
        self.max_rules = max_rules
        self.gamma = evict_percentile
        self.optimization_interval = optimization_interval
        self.warm_start_conf = warm_start_conf
        self.meta_rule_fn = meta_rule_fn
        self.verbose = verbose

        self.last_optimization_time = time.time()
        self.num_flipped = 0
        self.num_pruned = 0
        self.num_merged = 0
        self.num_evicted = 0

    # ------------------------------------------------------------- state tracking
    def update_rule_confidence(self, rule_id: str, sR: float):
        """Ingest one LLM contribution score (Section III-F, dynamic state tracking)."""
        rule = self.hypergraph.get_rule(rule_id)
        if rule is None:
            return
        rule.sR_sum += float(sR)
        if sR > 0:
            rule.used_count += 1
            rule.last_used_time = time.time()
        rule.update_confidence()
        rule.update_freshness(self.decay_lambda)

    def should_optimize(self) -> bool:
        return (time.time() - self.last_optimization_time) >= self.optimization_interval

    # ------------------------------------------------------------- optimisation
    def optimize(self) -> Dict:
        """Run the four periodic operations and return a summary."""
        self.last_optimization_time = time.time()
        if not self.hypergraph.rules:
            return {"pruned": 0, "flipped": 0, "merged": 0, "evicted": 0}

        pruned = self._prune_stale()
        flipped, re_pruned = self._flip_and_reevaluate()
        merged = self._merge_similar_rules()
        evicted = self._evict_low_utility_rules()

        self.num_pruned += pruned + re_pruned
        self.num_flipped += flipped
        self.num_merged += merged
        self.num_evicted += evicted

        summary = {"pruned": pruned + re_pruned, "flipped": flipped,
                   "merged": merged, "evicted": evicted}
        if self.verbose and any(summary.values()):
            print(f"[RuleMaintenance] {summary}")
        return summary

    def _prune_stale(self) -> int:
        """Fresh(R) < theta_f -> permanently purge (paper: 'due to staleness')."""
        stale = []
        for rule in list(self.hypergraph.rules.values()):
            rule.update_freshness(self.decay_lambda)
            if rule.fresh < self.theta_f:
                stale.append(rule.rule_id)
        for rid in stale:
            self._remove_rule(rid)
        return len(stale)

    def _flip_and_reevaluate(self):
        """Flip persistent low-confidence rules and prune those that do not recover.

        Paper: "we flip the rule's conclusion ... and reset Conf(R) to a
        warm-start value for re-evaluation. If Conf(R) remains below theta_c
        after N_eval additional triggers, the rule is permanently pruned."
        Only 1 -> 0 flips are performed, as stated in Section III-F.
        """
        flipped, re_pruned = 0, 0

        for rule in list(self.hypergraph.rules.values()):
            # (a) re-evaluation of a previously flipped rule
            if rule.flipped_at_trigger >= 0:
                extra = rule.trigger_count - rule.flipped_at_trigger
                if extra >= self.N_eval:
                    if rule.conf < self.theta_c:
                        self._remove_rule(rule.rule_id)
                        re_pruned += 1
                    else:
                        rule.flipped_at_trigger = -1
                continue

            # (b) fresh flip decision
            if (rule.conclusion == 1
                    and rule.conf < self.theta_c
                    and rule.trigger_count >= self.N_eval
                    and rule.used_count < rule.trigger_count * 0.3):
                rule.conclusion = 0
                rule.trigger_count = 0
                rule.sR_sum = 0.0
                rule.used_count = 0
                rule.conf0 = self.warm_start_conf
                rule.conf = self.warm_start_conf
                rule.flipped_at_trigger = 0
                flipped += 1

        return flipped, re_pruned

    # ------------------------------------------------------------- merging
    def _merge_similar_rules(self) -> int:
        """Cluster by atom-set Jaccard and consolidate each cluster via Mlight."""
        rules = list(self.hypergraph.rules.values())
        consumed: set = set()
        merged_count = 0

        for i in range(len(rules)):
            r1 = rules[i]
            if r1.rule_id in consumed:
                continue
            cluster = [r1]
            for j in range(i + 1, len(rules)):
                r2 = rules[j]
                if r2.rule_id in consumed or r2.conclusion != r1.conclusion:
                    continue
                a1, a2 = r1.atom_set(), r2.atom_set()
                if not a1 or not a2:
                    continue
                jaccard = len(a1 & a2) / len(a1 | a2)
                if jaccard >= self.eta_m:
                    cluster.append(r2)

            if len(cluster) < 2:
                continue

            meta = self._build_meta_rule(cluster)
            if meta is None:
                continue

            for r in cluster:
                consumed.add(r.rule_id)
            self._apply_meta_rule(meta, cluster)
            merged_count += len(cluster) - 1

        return merged_count

    def _build_meta_rule(self, cluster: List[FOLRule]) -> Optional[FOLRule]:
        """Eq. 10: Conf(R_c) = sum(Conf_i * |Trig_i|) / sum(|Trig_i|)."""
        if self.meta_rule_fn is not None:
            try:
                meta = self.meta_rule_fn(cluster)
                if meta is not None:
                    return meta
            except Exception as exc:
                if self.verbose:
                    print(f"[RuleMaintenance] Mlight meta-rule failed: {exc}")

        # Fallback: generalise by keeping only atoms shared by every constituent.
        shared = set.intersection(*[r.atom_set() for r in cluster]) if cluster else set()
        atoms: List[ConditionAtom] = []
        if shared:
            seen = set()
            for r in cluster:
                for atom in r.atoms:
                    if atom.to_key() in shared and atom.to_key() not in seen:
                        atoms.append(ConditionAtom(atom.atom_type, atom.attr,
                                                   atom.value1, atom.value2,
                                                   atom.threshold))
                        seen.add(atom.to_key())
        if not atoms:
            return None

        total_trig = sum(r.trigger_count for r in cluster)
        conf = (sum(r.conf * r.trigger_count for r in cluster) / total_trig
                if total_trig > 0 else
                sum(r.conf for r in cluster) / len(cluster))

        return FOLRule(
            rule_id=new_id("meta"),
            atoms=atoms,
            conclusion=cluster[0].conclusion,
            conf=conf,
            conf0=conf,
            trigger_count=total_trig,
            sR_sum=sum(r.sR_sum for r in cluster),
            used_count=sum(r.used_count for r in cluster),
            last_used_time=max(r.last_used_time for r in cluster),
            created_time=min(r.created_time for r in cluster),
        )

    def _apply_meta_rule(self, meta: FOLRule, cluster: List[FOLRule]):
        """Replace the constituents by the meta-rule throughout the hypergraph."""
        old_ids = {r.rule_id for r in cluster}
        affected = [he for he in self.hypergraph.hyperedges.values()
                    if any(r.rule_id in old_ids for r in he.rules)]

        for edge in affected:
            edge.rules = [r for r in edge.rules if r.rule_id not in old_ids]
            edge.rules.append(meta)
            for atom in meta.atoms:
                self.hypergraph.inverted_index[f"__attr__::{atom.attr.lower()}"].add(
                    edge.hyperedge_id)

        for rid in old_ids:
            self.hypergraph.rules.pop(rid, None)

        self.hypergraph.rules[meta.rule_id] = meta

    # ------------------------------------------------------------- forgetting
    def _evict_low_utility_rules(self) -> int:
        """Evict the bottom gamma percentile by Conf(R) * Fresh(R)."""
        if len(self.hypergraph.rules) <= self.max_rules:
            return 0

        rules = sorted(self.hypergraph.rules.values(),
                       key=lambda r: r.conf * r.fresh)
        n_evict = max(1, int(len(rules) * self.gamma))
        for rule in rules[:n_evict]:
            self._remove_rule(rule.rule_id)
        return n_evict

    def _remove_rule(self, rule_id: str):
        self.hypergraph.rules.pop(rule_id, None)
        for edge in self.hypergraph.hyperedges.values():
            edge.rules = [r for r in edge.rules if r.rule_id != rule_id]

    # ------------------------------------------------------------- stats
    def get_stats(self) -> Dict:
        rules = list(self.hypergraph.rules.values())
        confs = [r.conf for r in rules]
        return {
            "num_rules": len(rules),
            "mean_conf": sum(confs) / len(confs) if confs else 0.0,
            "num_flipped": self.num_flipped,
            "num_pruned": self.num_pruned,
            "num_merged": self.num_merged,
            "num_evicted": self.num_evicted,
        }