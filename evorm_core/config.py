#!/usr/bin/env python3
"""EvoRM configuration -- defaults follow Section IV-A of the paper."""

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


@dataclass
class EvoRMConfig:
    # ---------------------------------------------------------------- storage (III-D)
    alpha: float = 0.6                 # symbolic term of w_k        (Eq. 6)
    beta: float = 0.4                  # neural term of w_k          (Eq. 6), alpha + beta = 1
    merge_threshold: float = 0.5       # eta_h, hyperedge merge Jaccard (Eq. 7)
    K: int = 5                         # top-K hyperedges in E_sub
    demb: int = 1024                   # entity embedding dimension
    n_features: int = 8192             # hashing features for entity embeddings
    enable_attr_fallback: bool = True  # schema-level retrieval fallback (see docs)

    # ---------------------------------------------------------------- controller (III-E)
    theta_hi: float = 0.65             # direct-matching threshold
    theta_prune: float = 0.75          # direct non-matching threshold
    theta_gate: float = 0.3            # MLP gate threshold
    route_nonmatch: bool = True
    use_mlight: bool = True
    enable_mlp_gate: bool = True
    stage1_feedback: str = "decisive"  # how Stage-1 routes feed Eq. 8

    # ---------------------------------------------------------------- MLP gate
    mlp_hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    mlp_n_warmup: int = 500            # N_warmup, paper Section IV-A
    mlp_epochs: int = 50
    mlp_batch_size: int = 32
    mlp_learning_rate: float = 1e-3

    # ---------------------------------------------------------------- maintenance (III-F)
    decay_lambda: float = 0.01         # lambda in Eq. 9
    freshness_threshold: float = 0.1   # theta_f
    confidence_threshold: float = 0.2  # theta_c
    eval_triggers: int = 10            # N_eval
    merge_similarity: float = 0.7      # eta_m
    max_rules: int = 10000             # |R|_max
    evict_percentile: float = 0.1      # gamma
    optimization_interval: float = 300.0
    warm_start_conf: float = 0.3

    # ---------------------------------------------------------------- LLMs (IV-A)
    model: str = "meta-llama/Llama-3.1-8B-Instruct"
    mlight_model: str = "Qwen/Qwen2.5-3B-Instruct"
    temperature: float = 0.1
    max_retries: int = 3

    # ---------------------------------------------------------------- run control
    ablation_mode: Optional[str] = None   # None | no_stage1 | no_maintenance | no_hypergraph
    persistence_dir: Optional[str] = None

    # ---------------------------------------------------------------- helpers
    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "EvoRMConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "EvoRMConfig":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))


ABLATION_CONFIGS = {
    "full": {},
    "w/o U": {"ablation_mode": "no_hypergraph"},
    "w/o G": {"ablation_mode": "no_maintenance"},
    "w/o Stage-1": {"ablation_mode": "no_stage1"},
}


def ablation_config(name: str, base: Optional[EvoRMConfig] = None) -> EvoRMConfig:
    """Return one of the Table VI configurations."""
    base = base or EvoRMConfig()
    overrides = ABLATION_CONFIGS[name]
    d = base.to_dict()
    d.update(overrides)
    return EvoRMConfig.from_dict(d)