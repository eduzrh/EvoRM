"""EvoRM core: rule encoding, hypergraph storage, inference control, maintenance."""

from .config import EvoRMConfig
from .controller import TwoStageInferenceController
from .hypergraph import HypergraphStorage
from .maintenance import RuleMaintenance
from .plugin import EvoRMPlugin
from .rule_encoding import RuleEncoding
from .types import ConditionAtom, FOLRule, Hyperedge

__all__ = [
    "EvoRMConfig", "EvoRMPlugin", "RuleEncoding", "HypergraphStorage",
    "TwoStageInferenceController", "RuleMaintenance",
    "ConditionAtom", "FOLRule", "Hyperedge",
]
