#!/usr/bin/env python3
"""
MLP Gate g_phi -- paper Section III-E ("MLP gating") and Appendix I2.

Feature vector x_st (64 dims) -- exactly the three groups described in the paper:

    (i)   entity embeddings        (24 dims)
    (ii)  symbolic rule features   (24 dims)  rule-type distribution + Conf(R) statistics
    (iii) topological features     (16 dims)  hypergraph structural context of the nodes

Training is self-supervised: a surviving pair that the LLM resolves as MATCH
yields a positive sample, a NON-MATCH yields a negative sample. No human
annotation is required, matching Appendix I2.

A pair advances to Stage 2 only when g_phi(x_st) >= theta_gate; otherwise it is
directly assigned a negative decision.
"""

import threading
from typing import Dict, List, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    _TORCH_AVAILABLE = True
except ImportError:                                    # pragma: no cover
    torch, nn, optim = None, None, None
    _TORCH_AVAILABLE = False

EMB_DIM = 24
RULE_DIM = 24
TOPO_DIM = 16
FEATURE_DIM = EMB_DIM + RULE_DIM + TOPO_DIM           # 64

ATOM_TYPES = ("SameValue", "DifferValue", "SemanticEquiv",
              "SemanticConflict", "ShareNeighbor", "DifferNeighbor")


class _GateNet(nn.Module if _TORCH_AVAILABLE else object):
    def __init__(self, input_dim: int, hidden_dims: List[int]):
        super().__init__()
        layers, prev = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=False), nn.Dropout(0.2)]
            prev = h
        layers += [nn.Linear(prev, 1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MLPGate:
    """Lightweight gate predicting whether an ambiguous pair needs the LLM."""

    def __init__(self,
                 input_dim: int = FEATURE_DIM,
                 hidden_dims: Optional[List[int]] = None,
                 theta_gate: float = 0.3,
                 n_warmup: int = 500,
                 demb: int = 1024,
                 device: str = "cpu"):
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for the MLP gate: pip install torch")

        self.theta_gate = theta_gate
        self.n_warmup = n_warmup
        self.device = device
        self.demb = demb

        hidden_dims = hidden_dims or [256, 128, 64]
        self.net = _GateNet(input_dim, hidden_dims).to(device)

        self.is_trained = False
        self.samples_features: List[np.ndarray] = []
        self.samples_labels: List[float] = []
        self.trajectory_count = 0
        self._lock = threading.Lock()

        self.optimizer = None
        self.criterion = nn.BCELoss() if _TORCH_AVAILABLE else None

        self.total_predictions = 0
        self.llm_skipped = 0

        # Fixed random projection used to compress 2*demb -> 20 dims.
        self._proj = None
        self._proj_dim = EMB_DIM - 4

    # ------------------------------------------------------------- features
    def _projection(self, dim: int) -> np.ndarray:
        if self._proj is None or self._proj.shape[0] != dim:
            rng = np.random.RandomState(2024)
            self._proj = rng.normal(0.0, 1.0 / np.sqrt(dim), size=(dim, self._proj_dim))
            self._proj = self._proj.astype(np.float32)
        return self._proj

    def _embedding_features(self, es_emb, et_emb) -> List[float]:
        """(i) 24 dims derived from the entity embeddings e_s and e_t."""
        if es_emb is None or et_emb is None:
            return [0.0] * EMB_DIM

        a = np.asarray(es_emb, dtype=np.float32).ravel()
        b = np.asarray(et_emb, dtype=np.float32).ravel()
        if a.size != b.size or a.size == 0:
            return [0.0] * EMB_DIM

        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        denom = max(na * nb, 1e-8)
        cos = float(np.dot(a, b) / denom)
        dot = float(np.dot(a, b)) / 100.0
        l2 = float(np.linalg.norm(a - b)) / 100.0
        norm_ratio = min(na, nb) / max(na, nb, 1e-8)

        diff = (a - b) / max(na, nb, 1e-8)
        proj = diff @ self._projection(diff.size)
        return [cos, dot, l2, norm_ratio] + [float(v) for v in proj]

    def _rule_features(self, triggered_rules: List) -> List[float]:
        """(ii) 24 dims: rule-type distribution profile + Conf(R) statistics."""
        counts = {t: 0 for t in ATOM_TYPES}
        total_atoms = 0
        for rule in triggered_rules:
            for atom in getattr(rule, "atoms", []):
                atype = getattr(atom, "atom_type", "")
                if atype in counts:
                    counts[atype] += 1
                total_atoms += 1

        feats = [counts[t] / max(1, total_atoms) for t in ATOM_TYPES]   # 6

        if triggered_rules:
            confs = np.array([float(r.conf) for r in triggered_rules], dtype=np.float32)
            triggers = np.array([float(getattr(r, "trigger_count", 0))
                                 for r in triggered_rules], dtype=np.float32)
            n_match = sum(1 for r in triggered_rules if r.conclusion == 1)
            n_nonmatch = len(triggered_rules) - n_match

            feats += [
                n_match / len(triggered_rules),
                n_nonmatch / len(triggered_rules),
                float(confs.max()),                       # max historical confidence
                float(confs.mean()),
                float(confs.std()),
                float(confs.min()),
                float(np.log1p(triggers.max())),
                float(np.log1p(triggers.sum())),
                float(np.log1p(len(triggered_rules))),
                float(len(triggered_rules) > 0 and
                      (n_match == 0 or n_nonmatch == 0)),  # unanimity
                float(abs(
                    max([r.conf for r in triggered_rules if r.conclusion == 1], default=0.0)
                    - max([r.conf for r in triggered_rules if r.conclusion == 0], default=0.0))),
                float(np.mean([1.0 if getattr(r, "trigger_count", 0) > 0 else 0.0
                               for r in triggered_rules])),
                float(np.mean([getattr(r, "fresh", 1.0) for r in triggered_rules])),
            ]

        while len(feats) < RULE_DIM:
            feats.append(0.0)
        return feats[:RULE_DIM]

    def _topological_features(self, es_ctx: Dict, et_ctx: Dict,
                              es_emb, et_emb, hypergraph) -> List[float]:
        """(iii) 16 dims capturing the hypergraph structural context."""
        def rel_keys(ctx):
            return {k[len("neighbors_"):] for k in (ctx or {}) if k.startswith("neighbors_")}

        def neighbors(ctx, rel):
            v = (ctx or {}).get(f"neighbors_{rel}", set())
            return set(v) if isinstance(v, (set, list, tuple)) else set()

        es_rels, et_rels = rel_keys(es_ctx), rel_keys(et_ctx)
        shared_rels = es_rels & et_rels

        shared_entities = 0
        total_entities = 0
        for rel in shared_rels:
            n1, n2 = neighbors(es_ctx, rel), neighbors(et_ctx, rel)
            shared_entities += len(n1 & n2)
            total_entities += len(n1 | n2)

        feats = [
            float(len(es_rels)),
            float(len(et_rels)),
            float(len(shared_rels)),
            len(shared_rels) / max(1, len(es_rels | et_rels)),
            float(np.log1p(sum(len(neighbors(es_ctx, r)) for r in es_rels))),
            float(np.log1p(sum(len(neighbors(et_ctx, r)) for r in et_rels))),
            shared_entities / max(1, total_entities),
            float(np.log1p(shared_entities)),
        ]

        if hypergraph is not None:
            try:
                he_s = hypergraph.get_candidate_hyperedges([], [es_ctx])
                he_t = hypergraph.get_candidate_hyperedges([], [et_ctx])
                ids_t = {h.hyperedge_id for h in he_t}
                shared_edges = sum(1 for h in he_s if h.hyperedge_id in ids_t)
                feats += [float(np.log1p(len(he_s))), float(np.log1p(len(he_t))),
                          float(np.log1p(shared_edges)),
                          float(np.log1p(max(h.weight for h in he_s + he_t)))
                          if (he_s or he_t) else 0.0]
            except Exception:
                feats += [0.0, 0.0, 0.0, 0.0]
        else:
            feats += [0.0, 0.0, 0.0, 0.0]

        if es_emb is not None and et_emb is not None:
            feats.append(1.0)
        else:
            feats.append(0.0)

        while len(feats) < TOPO_DIM:
            feats.append(0.0)
        return feats[:TOPO_DIM]

    def extract_features(self,
                         es_context: Dict,
                         et_context: Dict,
                         triggered_rules: List,
                         es_embedding=None,
                         et_embedding=None,
                         hypergraph=None) -> np.ndarray:
        """Assemble the 64-dim gating feature vector x_st."""
        feats = (self._embedding_features(es_embedding, et_embedding)
                 + self._rule_features(triggered_rules or [])
                 + self._topological_features(es_context, et_context,
                                              es_embedding, et_embedding, hypergraph))
        return np.asarray(feats, dtype=np.float32)# ------------------------------------------------------------- inference
    def predict_proba(self, features: np.ndarray) -> float:
        with torch.no_grad():
            x = torch.from_numpy(np.asarray(features, dtype=np.float32)).unsqueeze(0).to(self.device)
            prob = float(self.net(x).item())
        self.total_predictions += 1
        return prob

    def should_invoke_llm(self, features: np.ndarray) -> bool:
        """True = forward to Stage 2; False = assign a negative decision directly."""
        if not self.is_trained:
            return True
        prob = self.predict_proba(features)
        if prob < self.theta_gate:
            self.llm_skipped += 1
            return False
        return True

    # ------------------------------------------------------------- training
    def collect_sample(self, features: np.ndarray, label: float):
        """Register one self-supervised sample (label from the Stage-2 decision)."""
        with self._lock:
            if len(self.samples_features) < 4 * self.n_warmup:
                self.samples_features.append(np.asarray(features, dtype=np.float32))
                self.samples_labels.append(float(label))
            self.trajectory_count += 1

    def is_ready_to_train(self) -> bool:
        return (not self.is_trained
                and self.trajectory_count >= self.n_warmup
                and len(set(self.samples_labels)) > 1
                and len(self.samples_features) >= 10)

    def fit_model(self, epochs: int = 50, batch_size: int = 32, verbose: bool = True):
        """Fit g_phi on the collected self-supervised samples."""
        if len(self.samples_features) < 10:
            if verbose:
                print(f"[MLPGate] not enough samples ({len(self.samples_features)})")
            return

        with self._lock:
            if self.is_trained:
                return
            self.is_trained = True          # reserve training slot

        X = torch.from_numpy(np.stack(self.samples_features)).float().to(self.device)
        y = torch.tensor(self.samples_labels).float().to(self.device)

        n_train = max(1, int(0.8 * len(X)))
        perm = torch.randperm(len(X))
        X_tr, y_tr = X[perm[:n_train]], y[perm[:n_train]]
        X_va, y_va = X[perm[n_train:]], y[perm[n_train:]]
        if len(X_va) == 0:
            X_va, y_va = X_tr, y_tr

        if self.optimizer is None:
            self.optimizer = optim.Adam(self.net.parameters(), lr=1e-3, weight_decay=1e-5)

        best_acc, best_state = 0.0, None
        for epoch in range(epochs):
            self.net.train()
            for i in range(0, len(X_tr), batch_size):
                bx, by = X_tr[i:i + batch_size], y_tr[i:i + batch_size]
                self.optimizer.zero_grad()
                loss = self.criterion(self.net(bx).squeeze(-1), by)
                loss.backward()
                self.optimizer.step()

            self.net.eval()
            with torch.no_grad():
                pred = (self.net(X_va).squeeze(-1) > 0.5).float()
                acc = float((pred == y_va).float().mean().item())
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}

        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.net.eval()
        self.is_trained = True

        if verbose:
            n_pos = int((y == 1).sum().item())
            print(f"[MLPGate] trained on {len(X)} samples "
                  f"(pos={n_pos}, neg={len(y) - n_pos}), val_acc={best_acc:.4f}")

    # ------------------------------------------------------------- persistence
    def save(self, path: str):
        import os
        state = {
            "model_state": self.net.state_dict(),
            "is_trained": self.is_trained,
            "trajectory_count": self.trajectory_count,
            "n_warmup": self.n_warmup,
            "theta_gate": self.theta_gate,
            "total_predictions": self.total_predictions,
            "llm_skipped": self.llm_skipped,
            "samples_features": self.samples_features,
            "samples_labels": self.samples_labels,
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(state, path)

    def load(self, path: str) -> bool:
        import os
        if not os.path.exists(path):
            return False
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.net.load_state_dict(state["model_state"])
        self.is_trained = bool(state.get("is_trained", False))
        self.trajectory_count = int(state.get("trajectory_count", 0))
        self.n_warmup = int(state.get("n_warmup", self.n_warmup))
        self.theta_gate = float(state.get("theta_gate", self.theta_gate))
        self.total_predictions = int(state.get("total_predictions", 0))
        self.llm_skipped = int(state.get("llm_skipped", 0))
        self.samples_features = list(state.get("samples_features", []))
        self.samples_labels = list(state.get("samples_labels", []))
        self.net.eval()
        return True

    # ------------------------------------------------------------- stats
    def get_stats(self) -> Dict:
        return {
            "is_trained": self.is_trained,
            "trajectory_count": self.trajectory_count,
            "n_samples": len(self.samples_features),
            "total_predictions": self.total_predictions,
            "llm_skipped": self.llm_skipped,
            "skip_rate": self.llm_skipped / max(1, self.total_predictions),
        }