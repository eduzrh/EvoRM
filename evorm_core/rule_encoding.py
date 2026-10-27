#!/usr/bin/env python3
"""
Rule Encoding -- paper Section III-C.

Converts an LLM inference trajectory (prediction y_hat + rationale) into an
explicit first-order logic rule R(e1, e2) <- AND_{a_i in A(R)} a_i(e1, e2).

Three steps, following the paper:
  Step 1  Rationale elicitation      -> Mlight with template T_enc  (evorm_core.mlight)
  Step 2  FOL rule generation        -> template-based parsing (regex) + LLM-based parsing
  Step 3  Rule confidence init       -> Eq. 4, warm-start from structurally overlapping rules
"""

import hashlib
import json
import re
import time
from typing import Dict, List, Optional, Sequence, Tuple

from .types import ConditionAtom, FOLRule


class RuleEncoding:
    """Distil LLM matching decisions into explicit FOL rules (Section III-C)."""

    # ---- Template-based parsing patterns (Appendix I) ------------------------
    ATTR_SAME_PATTERN = re.compile(
        r'(\w+)\s*=\s*same\s*(?:\(([^)]*)\))?', re.IGNORECASE)
    ATTR_DIFFER_PATTERN = re.compile(
        r'(\w+)\s*=\s*differ\s*(?:\(([^)]*)\))?', re.IGNORECASE)
    NEIGHBOR_PATTERN = re.compile(
        r'neighbor\s*:\s*(\w+)', re.IGNORECASE)
    NEIGHBOR_DIFFER_PATTERN = re.compile(
        r'neighbor\s*:\s*differ\s*(\w*)', re.IGNORECASE)

    def __init__(self, client=None, model: str = "gpt-3.5-turbo-1106",
                 enable_llm_parsing: bool = True, temperature: float = 0.0):
        self.client = client
        self.model = model
        self.enable_llm_parsing = enable_llm_parsing and client is not None
        self.temperature = temperature

    # ==================================================================
    # Step 2a: Template-based parsing
    # ==================================================================
    def parse_rationale(self, rationale: str, conclusion: int) -> List[ConditionAtom]:
        """Parse a rationale into typed condition atoms using deterministic regex.

        Mapping (paper Section III-C, "Template-based parsing"):
            attr=same        -> SameValue(e1.attr, e2.attr)
            attr=differ      -> DifferValue(e1.attr, e2.attr)
            neighbor:rel     -> ShareNeighbor(e1, e2, rel)
            neighbor:differ  -> DifferNeighbor(e1, e2, rel)
        """
        atoms: List[ConditionAtom] = []

        decisive_match = re.search(r'\[DECISIVE\](.*?)(?:\[SUPPORTING\]|$)', rationale, re.DOTALL)
        supporting_match = re.search(r'\[SUPPORTING\](.*?)$', rationale, re.DOTALL)
        decisive_section = decisive_match.group(1) if decisive_match else ""
        supporting_section = supporting_match.group(1) if supporting_match else ""
        all_text = decisive_section + " " + supporting_section

        # attr=same(...)
        for m in self.ATTR_SAME_PATTERN.finditer(all_text):
            attr = m.group(1).strip().lower()
            val_str = m.group(2).strip() if m.group(2) else ""
            atoms.append(ConditionAtom(
                atom_type="SameValue", attr=attr, value1=val_str, value2=val_str))

        # attr=differ(value1 vs value2)
        for m in self.ATTR_DIFFER_PATTERN.finditer(all_text):
            attr = m.group(1).strip().lower()
            val_str = m.group(2).strip() if m.group(2) else ""
            if " vs " in val_str:
                v1, v2 = [p.strip() for p in val_str.split(" vs ", 1)]
            elif " vs. " in val_str:
                v1, v2 = [p.strip() for p in val_str.split(" vs. ", 1)]
            else:
                v1, v2 = val_str, val_str
            atoms.append(ConditionAtom(
                atom_type="DifferValue", attr=attr, value1=v1, value2=v2))

        # neighbor:rel  /  neighbor:differ rel
        for m in self.NEIGHBOR_PATTERN.finditer(all_text):
            rel = m.group(1).strip().lower()
            if rel and 'differ' not in rel:
                atoms.append(ConditionAtom(atom_type="ShareNeighbor", attr=rel))

        for m in self.NEIGHBOR_DIFFER_PATTERN.finditer(all_text):
            rel = m.group(1).strip().lower()
            if rel:
                atoms.append(ConditionAtom(atom_type="DifferNeighbor", attr=rel))

        return self._dedupe(atoms)

    # ==================================================================
    # Step 2b: LLM-based parsing (paper: SemanticEquiv / SemanticConflict)
    # ==================================================================
    def parse_semantic_atoms(
        self,
        semantic_equiv: Optional[Sequence[Tuple]] = None,
        semantic_conflict: Optional[Sequence[Tuple]] = None,
    ) -> List[ConditionAtom]:
        """Build SemanticEquiv / SemanticConflict atoms from Mlight's JSON output.

        Mlight returns entries such as ("venue", "0.85", "<evidence>"), i.e.
        (attr, threshold, evidence). They are turned into atoms whose feasibility
        is decided at verification time by comparing the attribute similarity
        against the reported threshold.
        """
        atoms: List[ConditionAtom] = []

        def _as_threshold(value, default: float) -> float:
            try:
                thr = float(value)
            except (TypeError, ValueError):
                return default
            # Tolerate percentages such as "85".
            if thr > 1.0:
                thr = thr / 100.0
            return min(1.0, max(0.0, thr))

        for item in (semantic_equiv or []):
            attr = str(item[0]).strip().lower()
            if not attr:
                continue
            atoms.append(ConditionAtom(
                atom_type="SemanticEquiv", attr=attr,
                value1=str(item[2]) if len(item) > 2 else "",
                threshold=_as_threshold(item[1] if len(item) > 1 else None, 0.85)))

        for item in (semantic_conflict or []):
            attr = str(item[0]).strip().lower()
            if not attr:
                continue
            atoms.append(ConditionAtom(
                atom_type="SemanticConflict", attr=attr,
                value1=str(item[2]) if len(item) > 2 else "",
                threshold=_as_threshold(item[1] if len(item) > 1 else None, 0.20)))

        return self._dedupe(atoms)

    def parse_with_llm(self, rationale: str, es_context: Dict, et_context: Dict) -> List[ConditionAtom]:
        """Secondary Mlight invocation for rationales not covered by the templates.

        Returns SemanticEquiv / SemanticConflict atoms. Silently returns an empty
        list when no client is configured, so the deterministic path remains usable.
        """
        if not self.enable_llm_parsing:
            return []

        prompt = (
            "You are given a matching rationale and two entity records. Extract the "
            "implicit conditions that are NOT expressible as literal equality or "
            "inequality, i.e. fuzzy value equivalence and irreconcilable conflicts.\n\n"
            f"Rationale:\n{rationale}\n\n"
            f"Entity A: {json.dumps(es_context, ensure_ascii=False, default=str)}\n"
            f"Entity B: {json.dumps(et_context, ensure_ascii=False, default=str)}\n\n"
            "Respond with JSON only:\n"
            '{"semantic_equiv": [{"attr": "...", "threshold": 0.85}], '
            '"semantic_conflict": [{"attr": "...", "threshold": 0.20}]}\n'
            'Use an empty list when nothing applies.'
        )
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
            )
            text = resp.choices[0].message.content
            data = self._extract_json(text)
            if not data:
                return []
            equiv = [(d.get("attr", ""), d.get("threshold", 0.85), "")
                     for d in data.get("semantic_equiv", []) if d.get("attr")]
            conflict = [(d.get("attr", ""), d.get("threshold", 0.20), "")
                        for d in data.get("semantic_conflict", []) if d.get("attr")]
            return self.parse_semantic_atoms(equiv, conflict)
        except Exception:
            return []

    # ==================================================================
    # Step 2c: assembling the rule
    # ==================================================================
    def create_rule(
        self,
        rationale: str,
        conclusion: int,
        used_rules: Optional[List[Tuple[FOLRule, float]]] = None,
        es_context: Optional[Dict] = None,
        et_context: Optional[Dict] = None,
        semantic_equiv: Optional[Sequence[Tuple]] = None,
        semantic_conflict: Optional[Sequence[Tuple]] = None,
        allow_llm_parsing: bool = False,
    ) -> FOLRule:
        """Build a FOL rule from a rationale (Section III-C, Step 2 + Step 3).

        Note on Step 3: Eq. 4 sets Conf0(R) to the maximum contribution score of
        structurally overlapping rules when R_used is non-empty, and to 0
        otherwise. No floor is applied -- a rule earns confidence only through the
        closed-loop feedback of Eq. 8.
        """
        atoms = self.parse_rationale(rationale, conclusion)
        atoms += self.parse_semantic_atoms(semantic_equiv, semantic_conflict)

        if not atoms and allow_llm_parsing and es_context and et_context:
            atoms = self.parse_with_llm(rationale, es_context, et_context)

        # Generic fallback: when neither templates nor Mlight produced an atom,
        # materialise ALL matching / differing attributes as one conjunctive rule.
        # This keeps the rule specific instead of degenerating into a wildcard.
        if not atoms:
            atoms = self._context_atoms(conclusion, es_context, et_context)

        atoms = self._dedupe(atoms)

        rule = FOLRule(
            rule_id=self.stable_rule_id(atoms, conclusion),
            atoms=atoms,
            conclusion=int(conclusion),
            created_time=time.time(),
        )

        # ---- Step 3: rule confidence initialization (Eq. 4) -------------------
        rule.conf0 = self.initial_confidence(atoms, used_rules)
        rule.conf = rule.conf0
        # Fresh(R) is 1 at inception; t*_R left at 0 so Eq. 9 yields 1.0 until
        # the first positive contribution arrives.
        rule.fresh = 1.0
        return rule

    def initial_confidence(
        self,
        atoms: List[ConditionAtom],
        used_rules: Optional[List[Tuple[FOLRule, float]]] = None,
    ) -> float:
        """Eq. 4: Conf0(R) = max s_R' over R' with O(R, R') > 0, else 0."""
        if not used_rules:
            return 0.0

        atom_set = set(a.to_key() for a in atoms)
        best = 0.0
        for used_rule, sR in used_rules:
            other = set(a.to_key() for a in used_rule.atoms)
            if not atom_set or not other:
                continue
            union = len(atom_set | other)
            if union == 0:
                continue
            if len(atom_set & other) / union > 0.0:
                best = max(best, float(sR))
        return best

    # ==================================================================
    # Rule merging (Section III-F, Eq. 10)
    # ==================================================================
    def abstract_meta_rule(self, cluster: List[FOLRule],
                           client=None) -> Optional[FOLRule]:
        """Consolidate a rule cluster into one abstract meta-rule via Mlight.

        Mlight generalises specific literals into typed variables (``$NAME``,
        ``$DATE``, ...). Returns ``None`` when no client is available or the
        response cannot be parsed, so the caller can fall back to structural
        intersection.
        """
        if not cluster:
            return None
        client = client or self.client
        if client is None:
            return None

        lines = []
        for idx, rule in enumerate(cluster, 1):
            body = " \u2227 ".join(f"{a.atom_type}({a.attr})" for a in rule.atoms)
            verdict = "MATCH" if rule.conclusion == 1 else "NON-MATCH"
            lines.append(f"  R{idx}: {body} -> {verdict}")

        prompt = (
            "Consolidate the following cluster of first-order logic entity-matching "
            "rules into ONE abstract meta-rule by generalising their specific "
            "literals into typed variables.\n\n"
            "Constituent rules:\n" + "\n".join(lines) + "\n\n"
            "Atom types: SameValue, DifferValue, SemanticEquiv, SemanticConflict, "
            "ShareNeighbor, DifferNeighbor.\n"
            "Typed variables must start with '$', e.g. $NAME, $DATE, $CATEGORY, "
            "$LOCATION.\n\n"
            "Respond with JSON only:\n"
            '{"atoms": [{"atom": "SameValue", "attr": "$NAME"}], '
            '"conclusion": 1, "reason": "..."}'
        )

        try:
            resp = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                response_format={"type": "json_object"},
            )
            data = self._extract_json(resp.choices[0].message.content)
        except Exception:
            return None
        if not data:
            return None

        atoms: List[ConditionAtom] = []
        for item in data.get("atoms", []):
            atom_type = str(item.get("atom", item.get("atom_type", ""))).strip()
            attr = str(item.get("attr", "")).strip()
            if not atom_type or not attr:
                continue
            if atom_type not in ("SameValue", "DifferValue", "SemanticEquiv",
                                 "SemanticConflict", "ShareNeighbor", "DifferNeighbor"):
                continue
            atoms.append(ConditionAtom(
                atom_type=atom_type, attr=attr,
                threshold=float(item.get("threshold", 0.0) or 0.0)))
        atoms = self._dedupe(atoms)
        if not atoms:
            return None

        conclusion = int(data.get("conclusion", cluster[0].conclusion))

        # Eq. 10: frequency-weighted average of the constituent confidences.
        total_trig = sum(r.trigger_count for r in cluster)
        if total_trig > 0:
            conf = sum(r.conf * r.trigger_count for r in cluster) / total_trig
        else:
            conf = sum(r.conf for r in cluster) / len(cluster)

        return FOLRule(
            rule_id=self.stable_rule_id(atoms, conclusion),
            atoms=atoms,
            conclusion=conclusion,
            conf=conf,
            conf0=conf,
            trigger_count=total_trig,
            sR_sum=sum(r.sR_sum for r in cluster),
            used_count=sum(r.used_count for r in cluster),
            last_used_time=max(r.last_used_time for r in cluster),
            created_time=min(r.created_time for r in cluster),
        )

    def stable_rule_id(self, atoms: List[ConditionAtom],
                       conclusion: int = 1) -> str:
        """Deterministic rule ID from the body and the conclusion.

        The conclusion is part of the key: a positive and a negative rule with
        the same body are different rules and must not collide.
        """
        key = "|".join(sorted(a.to_key() for a in atoms)) + f"|y={int(conclusion)}"
        return hashlib.md5(key.encode()).hexdigest()[:16]

    # ==================================================================
    # Helpers
    # ==================================================================
    def _context_atoms(self, conclusion: int, es_context: Optional[Dict],
                       et_context: Optional[Dict]) -> List[ConditionAtom]:
        atoms: List[ConditionAtom] = []
        if not es_context or not et_context:
            return [ConditionAtom(
                atom_type="SameValue" if conclusion == 1 else "DifferValue",
                attr="entity_name")]

        keys = list(dict.fromkeys(list(es_context.keys()) + list(et_context.keys())))
        for key in keys:
            if key.startswith('neighbors_'):
                continue
            es_val = str(es_context.get(key, '')).strip()
            et_val = str(et_context.get(key, '')).strip()
            if not es_val or not et_val:
                continue
            same = es_val.lower() == et_val.lower()
            if conclusion == 1 and same:
                atoms.append(ConditionAtom(atom_type="SameValue", attr=key))
            elif conclusion == 0 and not same:
                atoms.append(ConditionAtom(atom_type="DifferValue", attr=key))

        if not atoms:
            atoms.append(ConditionAtom(
                atom_type="SameValue" if conclusion == 1 else "DifferValue",
                attr="entity_name"))
        return atoms

    @staticmethod
    def _dedupe(atoms: List[ConditionAtom]) -> List[ConditionAtom]:
        seen, out = set(), []
        for a in atoms:
            if a.to_key() in seen:
                continue
            seen.add(a.to_key())
            out.append(a)
        return out

    @staticmethod
    def _extract_json(text: str) -> Optional[Dict]:
        if not text:
            return None
        fenced = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
        candidate = fenced.group(1) if fenced else text
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            brace = re.search(r'\{.*\}', candidate, re.DOTALL)
            if brace:
                try:
                    return json.loads(brace.group(0))
                except json.JSONDecodeError:
                    return None
        return None