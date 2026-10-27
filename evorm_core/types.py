#!/usr/bin/env python3
"""
Core data structures for EvoRM.

Maps to the paper's Table I notations:

    ConditionAtom  -> a_i(e1, e2)   a condition atom of a FOL rule
    FOLRule        -> R             a first-order logic rule
    Hyperedge      -> H_k = (M_k, V_k, R_k, w_k)

Reference: EvoRM, Section III-C (Rule Encoding) and III-D (Hypergraph Storage).
"""

from dataclasses import dataclass, field
import math
import time
import uuid
from typing import Dict, List, Optional, Set, Tuple


# ------------------------------------------------------------------------------
# Atom types (paper Section III-C, "Template-based parsing" + "LLM-based parsing")
# ------------------------------------------------------------------------------
DATA_DEPENDENT_ATOMS = ("SameValue", "DifferValue", "SemanticEquiv", "SemanticConflict")
GRAPH_DEPENDENT_ATOMS = ("ShareNeighbor", "DifferNeighbor")
ATOM_TYPES = DATA_DEPENDENT_ATOMS + GRAPH_DEPENDENT_ATOMS


@dataclass
class ConditionAtom:
    """A single condition atom a_i(e1, e2) in a FOL rule.

    Attributes:
        atom_type: one of ATOM_TYPES.
        attr:      attribute name (data-dependent) or relation name (graph-dependent).
        value1:    literal observed on the source entity (optional, for provenance).
        value2:    literal observed on the target entity (optional, for provenance).
        threshold: similarity threshold for SemanticEquiv / SemanticConflict atoms.
                   Ignored for literal atoms.
    """

    atom_type: str
    attr: str
    value1: str = ""
    value2: str = ""
    threshold: float = 0.0

    def to_key(self) -> str:
        """Structural identity of the atom, used for Jaccard similarity O(R, R')."""
        return f"{self.atom_type}({self.attr})"

    def __hash__(self):
        return hash(self.to_key())

    def __eq__(self, other):
        return isinstance(other, ConditionAtom) and self.to_key() == other.to_key()


@dataclass
class FOLRule:
    """A first-order logic rule R distilled from an LLM inference trajectory.

    R(e1, e2) <- AND_{a_i in A(R)} a_i(e1, e2),  with a binary conclusion y_R.

    Confidence is maintained exactly as in Eq. 8 of the paper:

        Conf(R) = (Conf0(R) + sum_tau s_R(tau)) / (1 + sum_tau I[R in R_trig(tau)])

    where `conf0` is Conf0(R), `sR_sum` accumulates sum_tau s_R(tau) and
    `trigger_count` accumulates sum_tau I[R in R_trig(tau)].
    """

    rule_id: str
    atoms: List[ConditionAtom]
    conclusion: int              # y_R: 1 = match, 0 = non-match
    conf: float = 0.0            # Conf(R)
    conf0: float = 0.0           # Conf0(R), Eq. 4
    fresh: float = 1.0           # Fresh(R), Eq. 9
    trigger_count: int = 0       # |Trig(R)|
    used_count: int = 0          # times the LLM actually used this rule (s_R > 0)
    sR_sum: float = 0.0          # sum of contribution scores
    last_used_time: float = 0.0  # t*_R: last trajectory with s_R > 0
    created_time: float = 0.0
    # Bookkeeping for the "flip then re-evaluate over N_eval triggers" procedure
    # of Section III-F. -1 means "never flipped".
    flipped_at_trigger: int = -1

    # ------------------------------------------------------------------ Eq. 8
    def update_confidence(self):
        """Conf(R) = (Conf0(R) + sum sR) / (1 + |Trig(R)|)."""
        self.conf = (self.conf0 + self.sR_sum) / (1.0 + self.trigger_count)

    # ------------------------------------------------------------------ Eq. 9
    def update_freshness(self, decay_lambda: float = 0.01):
        """Fresh(R) = exp(-lambda * (t_now - t*_R))."""
        if self.last_used_time > 0:
            self.fresh = math.exp(-decay_lambda * (time.time() - self.last_used_time))
        else:
            self.fresh = 1.0

    # ------------------------------------------------------------- structural
    def atom_set(self) -> Set[str]:
        """A(R) as a set of structural keys, used for Jaccard O(R, R')."""
        return set(a.to_key() for a in self.atoms)

    def atoms_str(self) -> str:
        return " AND ".join(
            f"{a.atom_type}({a.attr})" for a in self.atoms
        ) if self.atoms else "TRUE"

    def verdict_str(self) -> str:
        return "MATCH" if self.conclusion == 1 else "NON-MATCH"

    def to_dict(self) -> Dict:
        return {
            "rule_id": self.rule_id,
            "atoms": [
                {
                    "atom_type": a.atom_type,
                    "attr": a.attr,
                    "value1": a.value1,
                    "value2": a.value2,
                    "threshold": a.threshold,
                }
                for a in self.atoms
            ],
            "conclusion": self.conclusion,
            "conf": self.conf,
            "conf0": self.conf0,
            "fresh": self.fresh,
            "trigger_count": self.trigger_count,
            "used_count": self.used_count,
            "sR_sum": self.sR_sum,
            "last_used_time": self.last_used_time,
            "created_time": self.created_time,
            "flipped_at_trigger": self.flipped_at_trigger,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "FOLRule":
        atoms = [
            ConditionAtom(
                atom_type=a.get("atom_type", ""),
                attr=a.get("attr", ""),
                value1=a.get("value1", ""),
                value2=a.get("value2", ""),
                threshold=float(a.get("threshold", 0.0) or 0.0),
            )
            for a in d.get("atoms", [])
        ]
        return cls(
            rule_id=d["rule_id"],
            atoms=atoms,
            conclusion=int(d.get("conclusion", 0)),
            conf=float(d.get("conf", 0.0)),
            conf0=float(d.get("conf0", 0.0)),
            fresh=float(d.get("fresh", 1.0)),
            trigger_count=int(d.get("trigger_count", 0)),
            used_count=int(d.get("used_count", 0)),
            sR_sum=float(d.get("sR_sum", 0.0)),
            last_used_time=float(d.get("last_used_time", 0.0)),
            created_time=float(d.get("created_time", 0.0)),
            flipped_at_trigger=int(d.get("flipped_at_trigger", -1)),
        )


@dataclass
class Hyperedge:
    """A matching cluster H_k = (M_k, V_k, R_k, w_k) of the hypergraph H_t."""

    hyperedge_id: str
    entity_pairs: Set[Tuple[int, int]] = field(default_factory=set)  # M_k
    node_set: Set[str] = field(default_factory=set)                  # V_k
    rules: List[FOLRule] = field(default_factory=list)               # R_k
    weight: float = 0.0                                              # w_k

    # ------------------------------------------------------------------ Eq. 6
    def recompute_weight(
        self,
        alpha: float = 0.6,
        beta: float = 0.4,
        entity_embeddings: Optional[Dict] = None,
        similarity_fn=None,
    ) -> float:
        """w_k = alpha * max_{R in R_k} Conf(R) + beta * (1/|M_k|) sum_{(ei,ej) in M_k} cos(ei, ej).

        The neural term is averaged over the *entity pairs* stored in the cluster
        (not over all stored embeddings), exactly as stated in Eq. 6.
        """
        symbolic = max((r.conf for r in self.rules), default=0.0)

        neural = 0.0
        if entity_embeddings and self.entity_pairs and similarity_fn is not None:
            sims = []
            for (eid1, eid2) in self.entity_pairs:
                v1 = entity_embeddings.get(f"e_{eid1}")
                v2 = entity_embeddings.get(f"e_{eid2}")
                if v1 is not None and v2 is not None:
                    sims.append(float(similarity_fn(v1, v2)))
            if sims:
                neural = sum(sims) / len(sims)

        self.weight = alpha * symbolic + beta * neural
        return self.weight


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"