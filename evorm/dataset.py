#!/usr/bin/env python3
"""
Minimal dataset loader.

EvoRM does not ship datasets. Convert each benchmark into JSONL with one of the
two schemas below and point the experiment scripts at the file.

Binary task (ER / SM / EA candidate confirmation):
    {"es_id": "r1", "et_id": "r2",
     "es_context": {"name": "...", "price": "..."},
     "et_context": {"name": "...", "price": "..."},
     "label": 1}

Ranking task (EA / EL):
    {"query_id": "q1",
     "query_context": {"name": "...", ...},
     "candidates": [{"entity_id": "c1", "context": {...}, "base_score": 0.72},
                    {"entity_id": "c2", "context": {...}, "base_score": 0.31}],
     "gold_id": "c1"}

Contexts may contain the key ``neighbors_<relation>`` whose value is a list of
neighbouring entity names; these feed the graph-dependent atoms
(ShareNeighbor / DifferNeighbor).
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union


@dataclass
class PairTask:
    es_id: str
    et_id: str
    es_context: Dict
    et_context: Dict
    label: int


@dataclass
class Candidate:
    entity_id: str
    context: Dict
    base_score: float = 0.0
    is_gold: bool = False


@dataclass
class RankingTask:
    query_id: str
    query_context: Dict
    candidates: List[Candidate] = field(default_factory=list)
    gold_id: Optional[str] = None


Task = Union[PairTask, RankingTask]


def load_tasks(path: str, limit: Optional[int] = None) -> List[Task]:
    tasks: List[Task] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            tasks.append(_parse(raw))
            if limit is not None and len(tasks) >= limit:
                break
    return tasks


def _parse(raw: Dict) -> Task:
    if "candidates" in raw:
        cands = [
            Candidate(
                entity_id=str(c.get("entity_id")),
                context=c.get("context", {}) or {},
                base_score=float(c.get("base_score", 0.0)),
                is_gold=bool(c.get("is_gold", False)) or c.get("entity_id") == raw.get("gold_id"),
            )
            for c in raw["candidates"]
        ]
        return RankingTask(
            query_id=str(raw["query_id"]),
            query_context=raw.get("query_context", {}) or {},
            candidates=cands,
            gold_id=str(raw["gold_id"]) if raw.get("gold_id") is not None else None,
        )

    return PairTask(
        es_id=str(raw["es_id"]),
        et_id=str(raw["et_id"]),
        es_context=raw.get("es_context", {}) or {},
        et_context=raw.get("et_context", {}) or {},
        label=int(raw.get("label", 0)),
    )


def is_ranking(tasks: List[Task]) -> bool:
    return bool(tasks) and isinstance(tasks[0], RankingTask)