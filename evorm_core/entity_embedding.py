#!/usr/bin/env python3
"""
EvoRM Entity Embedding Module
==============================
Paper Section III-D: Entity embeddings (demb=1024) for computing neural
cosine similarity between entity pairs within hyperedges.

The hyperedge weight wk combines:
  wk = α * max(Conf(R)) + β * avg_cos_sim(Vk)     [Eq. 6]

Where avg_cos_sim is the average cosine similarity between entity embeddings
of all entity pairs in the hyperedge.

Implementation:
  Uses feature hashing (the "hashing trick") to produce deterministic 1024-dim
  embeddings from entity attributes, neighbors, and descriptions. This is a
  lightweight, API-free approach that captures the structural information of
  entities without requiring external embedding services.
"""

import hashlib
import math
import re
from typing import Dict, List, Set, Optional, Tuple
import numpy as np


class EntityEmbedder:
    """
    Compute 1024-dimensional entity embeddings using feature hashing.
    
    Usage:
        embedder = EntityEmbedder(demb=1024)
        emb = embedder.embed_entity(entity_context)
        sim = embedder.cosine_similarity(emb1, emb2)
    """
    
    def __init__(self, demb: int = 1024, n_features: int = 8192):
        """
        Args:
            demb: Output embedding dimension (paper: 1024)
            n_features: Number of hash features for the hashing trick
        """
        self.demb = demb
        self.n_features = n_features
        self._cache: Dict[str, np.ndarray] = {}
        self._proj: Optional[np.ndarray] = None
    
    def _serialize_entity(self, context: Dict) -> str:
        """Serialize entity context into a canonical string for hashing."""
        parts = []
        
        # Sort keys for deterministic output
        for key in sorted(context.keys()):
            val = context[key]
            if key.startswith('neighbors_'):
                rel = key.replace('neighbors_', '')
                if isinstance(val, (set, list)):
                    items = sorted(str(v) for v in val)
                    for item in items:
                        parts.append(f"n:{rel}:{item}")
                else:
                    parts.append(f"n:{rel}:{val}")
            elif key == 'entity_name':
                parts.append(f"name:{val}")
            elif key == 'description':
                if val:
                    # Tokenize and add individual tokens
                    tokens = re.findall(r'\w+', str(val).lower())
                    for tok in tokens:
                        parts.append(f"desc:{tok}")
            else:
                parts.append(f"attr:{key}:{val}")
        
        return '|'.join(parts)
    
    def _feature_hash(self, text: str) -> np.ndarray:
        """
        Feature hashing: map text features to a fixed-size vector.
        
        Uses multiple hash functions to create a dense representation
        from sparse text features.
        """
        vec = np.zeros(self.n_features, dtype=np.float32)
        
        # Extract features (character n-grams + word tokens)
        features = []
        
        # Word-level tokens
        tokens = re.findall(r'\w+', text.lower())
        features.extend(tokens)
        
        # Character trigrams
        for i in range(len(text) - 2):
            features.append(f"c3:{text[i:i+3]}")
        
        # Bigrams of tokens
        for i in range(len(tokens) - 1):
            features.append(f"b2:{tokens[i]}_{tokens[i+1]}")
        
        # Hash each feature to a position in the vector
        for feat in features:
            h = hashlib.md5(feat.encode('utf-8')).hexdigest()
            # Use first 8 hex chars as two 4-hex indices
            idx1 = int(h[:4], 16) % self.n_features
            idx2 = int(h[4:8], 16) % self.n_features
            
            # Add +1 at idx1, -1 at idx2 (signed hashing)
            vec[idx1] += 1.0
            vec[idx2] -= 1.0
        
        # Normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        
        # Project to the target dimension with a fixed random matrix, built
        # once and reused for every entity.
        if self.n_features != self.demb:
            if self._proj is None:
                rng = np.random.default_rng(20240601)
                self._proj = (
                    rng.standard_normal((self.demb, self.n_features))
                    / math.sqrt(self.demb)
                ).astype(np.float32)
            vec = self._proj @ vec
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm

        return vec
    
    def embed_entity(self, context: Dict, entity_id: str = None) -> np.ndarray:
        """
        Compute entity embedding from context.
        
        Args:
            context: Entity context dict (attributes, neighbors, descriptions)
            entity_id: Optional entity ID for caching
            
        Returns:
            1024-dim normalized embedding vector
        """
        # Check cache
        cache_key = entity_id if entity_id else self._serialize_entity(context)
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Serialize and hash
        text = self._serialize_entity(context)
        emb = self._feature_hash(text)
        
        # Cache
        if len(self._cache) < 100000:  # Limit cache size
            self._cache[cache_key] = emb
        
        return emb
    
    def embed_batch(self, contexts: List[Dict], entity_ids: List[str] = None) -> np.ndarray:
        """Compute embeddings for a batch of entities."""
        entity_ids = entity_ids or [None] * len(contexts)
        embeddings = []
        for ctx, eid in zip(contexts, entity_ids):
            embeddings.append(self.embed_entity(ctx, eid))
        return np.stack(embeddings)
    
    def cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        dot = np.dot(emb1, emb2)
        return float(max(0.0, min(1.0, dot)))  # Already normalized, clamp for safety
    
    def pairwise_similarities(self, embeddings: List[np.ndarray]) -> List[float]:
        """
        Compute pairwise cosine similarities for all pairs in a list.
        
        Args:
            embeddings: List of entity embeddings
            
        Returns:
            List of cosine similarities for all unique pairs
        """
        n = len(embeddings)
        if n < 2:
            return []
        
        sims = []
        for i in range(n):
            for j in range(i + 1, n):
                sims.append(self.cosine_similarity(embeddings[i], embeddings[j]))
        return sims
    
    def avg_cosine_similarity(self, embeddings: List[np.ndarray]) -> float:
        """
        Compute average pairwise cosine similarity.
        
        This is the avg_cos_sim(Vk) term in Eq. 6 of the paper.
        
        Args:
            embeddings: List of entity embeddings in the hyperedge
            
        Returns:
            Average cosine similarity (0.0 if < 2 embeddings)
        """
        sims = self.pairwise_similarities(embeddings)
        if not sims:
            return 0.0
        return float(np.mean(sims))
    
    def clear_cache(self):
        """Clear the embedding cache."""
        self._cache.clear()
    
    def get_stats(self) -> Dict:
        return {
            'cache_size': len(self._cache),
            'demb': self.demb,
            'n_features': self.n_features,
        }


# ==============================================================================
# Test / Demo
# ==============================================================================

if __name__ == "__main__":
    print("EvoRM Entity Embedding - Self Test")
    print("=" * 60)
    
    embedder = EntityEmbedder(demb=1024, n_features=8192)
    
    # Test entities
    ctx1 = {
        'entity_name': 'Machine Learning Basics',
        'title': 'Machine Learning Basics',
        'year': '2020',
        'authors': 'Smith et al.',
        'neighbors_cites': {'Deep Learning Review', 'NN Basics'},
    }
    ctx2 = {
        'entity_name': 'Machine Learning Fundamentals',
        'title': 'Machine Learning Fundamentals',
        'year': '2020',
        'authors': 'Smith and Jones',
        'neighbors_cites': {'Deep Learning Review'},
    }
    ctx3 = {
        'entity_name': 'Quantum Physics Textbook',
        'title': 'Quantum Physics Textbook',
        'year': '2015',
        'authors': 'Einstein et al.',
        'neighbors_cites': {'Relativity Theory'},
    }
    
    # Compute embeddings
    emb1 = embedder.embed_entity(ctx1, 'e1')
    emb2 = embedder.embed_entity(ctx2, 'e2')
    emb3 = embedder.embed_entity(ctx3, 'e3')
    
    print(f"\n1. Embedding dimensions: {emb1.shape}")
    print(f"   Norms: {np.linalg.norm(emb1):.4f}, {np.linalg.norm(emb2):.4f}, {np.linalg.norm(emb3):.4f}")
    
    # Similarities
    sim_12 = embedder.cosine_similarity(emb1, emb2)
    sim_13 = embedder.cosine_similarity(emb1, emb3)
    sim_23 = embedder.cosine_similarity(emb2, emb3)
    
    print(f"\n2. Cosine similarities:")
    print(f"   sim(E1, E2) = {sim_12:.4f}  (similar ML books)")
    print(f"   sim(E1, E3) = {sim_13:.4f}  (ML vs Quantum)")
    print(f"   sim(E2, E3) = {sim_23:.4f}  (ML vs Quantum)")
    
    # Check that similar entities have higher similarity
    assert sim_12 > sim_13, f"Expected sim(E1,E2) > sim(E1,E3), got {sim_12:.4f} <= {sim_13:.4f}"
    assert sim_12 > sim_23, f"Expected sim(E1,E2) > sim(E2,E3), got {sim_12:.4f} <= {sim_23:.4f}"
    print("   ✅ Similar entities have higher similarity than dissimilar ones")
    
    # Average pairwise similarity
    avg_sim = embedder.avg_cosine_similarity([emb1, emb2, emb3])
    print(f"\n3. avg_cos_sim(Vk) for 3 entities: {avg_sim:.4f}")
    
    # Cache test
    emb1_cached = embedder.embed_entity(ctx1, 'e1')
    assert np.array_equal(emb1, emb1_cached), "Cache test failed"
    print(f"\n4. Cache: {embedder.get_stats()}")
    
    print("\n✅ All tests passed!")
