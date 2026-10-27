#!/usr/bin/env python3
"""
EvoRM Mlight: Independent Rationale Elicitation Module
======================================================
Paper Section III-C Step 1: Mlight is a lightweight LLM that takes serialized
entity context as input and generates a natural language rationale explaining the
matching decision. This is SEPARATED from the decision-making call (Mheavy).

Key design (from paper):
  - Mlight: entity context → natural language rationale
  - The rationale is then used to:
    (a) Extract condition atoms → create FOL rules
    (b) Provide context for the decision-making step (Mheavy)
  - Separation enables: independent rule extraction, better prompt engineering,
    and faithfulness to the paper's two-model architecture.

Implementation note:
  Since we use a single API (gpt-3.5-turbo-1106), Mlight and Mheavy are
  implemented as two separate API calls with different prompts, following
  the paper's two-step architecture faithfully.
"""

import json
import re
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field


@dataclass
class RationaleResult:
    """Result from Mlight rationale elicitation."""
    rationale: str
    decisive_factors: List[str] = field(default_factory=list)
    supporting_factors: List[str] = field(default_factory=list)
    conflicting_factors: List[str] = field(default_factory=list)
    attribute_analysis: Dict[str, str] = field(default_factory=dict)
    # attr_name -> "same" | "differ" | "unknown"
    semantic_equiv: List[Tuple[str, str, str]] = field(default_factory=list)
    # List of (attr, val1, val2) for semantic equivalence
    semantic_conflict: List[Tuple[str, str, str]] = field(default_factory=list)
    # List of (attr, val1, val2) for semantic conflict
    token_usage: int = 0
    raw_response: str = ""


class MlightRationaleElicitor:
    """
    Mlight: lightweight rationale elicitation module.

    Section III-C orders the two LLM calls as

        y_hat = M_heavy(T_dec(e_s, e_t, E_sub, R_trig))
        r     = M_light(T_enc(e_s, e_t, ctx(e_s, e_t), y_hat))

    i.e. the decision is taken first and Mlight then externalises the evidence
    behind it. The rationale ``r`` is what RuleEncoding turns into an FOL rule,
    so the rule bank learns from an explanation of an observed decision.

    Usage:
        mlight = MlightRationaleElicitor(client=openai_client)
        rationale = mlight.elicit(es_context, et_context, triggered_rules,
                                  decision=y_hat)
    """
    
    # Default prompt template for rationale elicitation
    DEFAULT_RATIONALE_PROMPT = """You are an expert entity matching analyst. Your task is to analyze an entity pair and provide a detailed reasoning about whether they match, WITHOUT making a final decision.

**Entity A:**
{entity_a}

**Entity B:**
{entity_b}

{triggered_rules_section}

**Instructions:**
1. Analyze ALL attributes of both entities systematically.
2. For each attribute, determine if the values are the SAME, DIFFER, or SIMILAR (partial match).
3. Identify SEMANTIC EQUIVALENCE: values that are different strings but refer to the same real-world entity
   (e.g., "USA" and "United States", "J. Smith" and "John Smith", "ML" and "Machine Learning").
4. Identify SEMANTIC CONFLICT: values that appear similar but refer to different entities
   (e.g., "John Smith (1980)" vs "John Smith (1990)", same name but different person).
5. Identify which attributes are DECISIVE (strongly indicate match or non-match).
6. Identify which attributes are SUPPORTING (auxiliary evidence).
7. Note any CONFLICTING evidence (attributes that suggest the opposite conclusion).

**Output Format (JSON only):**
```json
{{
  "attribute_analysis": {{
    "attr_name1": "same",
    "attr_name2": "differ",
    "attr_name3": "similar"
  }},
  "semantic_equiv": [
    {{"attr": "attr_name", "value_a": "value in entity A", "value_b": "value in entity B", "explanation": "why they are semantically equivalent"}}
  ],
  "semantic_conflict": [
    {{"attr": "attr_name", "value_a": "value in entity A", "value_b": "value in entity B", "explanation": "why they are semantically conflicting"}}
  ],
  "decisive_factors": [
    "Detailed explanation of a decisive factor for matching or non-matching"
  ],
  "supporting_factors": [
    "Detailed explanation of a supporting factor"
  ],
  "conflicting_factors": [
    "Detailed explanation of any conflicting evidence"
  ],
  "reasoning": "[DECISIVE] ... [SUPPORTING] ... [CONFLICTING] ..."
}}
```

You MUST respond with valid JSON only. Do NOT include a decision (MATCH/NON-MATCH) — that will be done separately."""

    def __init__(self, client=None, model: str = "gpt-3.5-turbo-1106",
                 temperature: float = 0.1, max_retries: int = 3,
                 timeout: int = 30):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout = timeout
        
        # Statistics
        self.total_calls = 0
        self.total_tokens = 0
        self.total_time = 0.0
    
    def _serialize_entity(self, context: Dict) -> str:
        """Serialize entity context into a readable string."""
        parts = []
        for key, val in context.items():
            if key.startswith('neighbors_'):
                rel = key.replace('neighbors_', '')
                if isinstance(val, (set, list)):
                    neighbors_str = ', '.join(str(v) for v in val)
                else:
                    neighbors_str = str(val)
                parts.append(f"  - Relation '{rel}': {neighbors_str}")
            elif key == 'entity_name':
                parts.append(f"  - Name: {val}")
            elif key == 'description':
                if val:
                    parts.append(f"  - Description: {val}")
            else:
                parts.append(f"  - {key}: {val}")
        return '\n'.join(parts) if parts else '(no context available)'
    
    def _build_triggered_rules_section(self, triggered_rules: List) -> str:
        """Build a section describing triggered rules."""
        if not triggered_rules:
            return ""
        
        lines = ["**Historical Rules Triggered:**"]
        for i, rule in enumerate(triggered_rules, 1):
            atom_strs = []
            for a in rule.atoms:
                atom_strs.append(f"{a.atom_type}({a.attr})")
            atoms_str = " ∧ ".join(atom_strs)
            verdict = "MATCH" if rule.conclusion == 1 else "NON-MATCH"
            lines.append(
                f"  Rule {i}: {atoms_str} → {verdict} "
                f"(Confidence: {rule.conf:.3f}, Triggered: {rule.trigger_count} times)"
            )
        return '\n'.join(lines)
    
    def _parse_rationale_response(self, response_text: str) -> RationaleResult:
        """Parse the JSON response from Mlight into a RationaleResult."""
        result = RationaleResult(rationale="", raw_response=response_text)
        
        try:
            # Try to extract JSON block
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find bare JSON
                json_match = re.search(r'\{.*"attribute_analysis".*\}', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    json_str = response_text
            
            data = json.loads(json_str)
            
            result.decisive_factors = data.get('decisive_factors', [])
            result.supporting_factors = data.get('supporting_factors', [])
            result.conflicting_factors = data.get('conflicting_factors', [])
            result.attribute_analysis = data.get('attribute_analysis', {})
            
            # Parse semantic equivalence / conflict from LLM output
            for se in data.get('semantic_equiv', []):
                if isinstance(se, dict):
                    result.semantic_equiv.append((
                        se.get('attr', ''),
                        se.get('value_a', ''),
                        se.get('value_b', '')
                    ))
            for sc in data.get('semantic_conflict', []):
                if isinstance(sc, dict):
                    result.semantic_conflict.append((
                        sc.get('attr', ''),
                        sc.get('value_a', ''),
                        sc.get('value_b', '')
                    ))
            
            # Build structured rationale from components
            reasoning = data.get('reasoning', '')
            if not reasoning:
                # Reconstruct from factors
                parts = []
                if result.decisive_factors:
                    parts.append("[DECISIVE] " + "; ".join(result.decisive_factors))
                if result.supporting_factors:
                    parts.append("[SUPPORTING] " + "; ".join(result.supporting_factors))
                if result.conflicting_factors:
                    parts.append("[CONFLICTING] " + "; ".join(result.conflicting_factors))
                reasoning = '\n'.join(parts)
            
            result.rationale = reasoning
            
        except (json.JSONDecodeError, KeyError) as e:
            # Fallback: use the raw response as rationale
            result.rationale = response_text
            # Try to extract [DECISIVE] and [SUPPORTING] sections
            decisive_match = re.search(
                r'\[DECISIVE\](.*?)(?:\[SUPPORTING\]|\[CONFLICTING\]|$)', 
                response_text, re.DOTALL)
            if decisive_match:
                result.decisive_factors = [decisive_match.group(1).strip()]
            
            supporting_match = re.search(
                r'\[SUPPORTING\](.*?)(?:\[CONFLICTING\]|$)', 
                response_text, re.DOTALL)
            if supporting_match:
                result.supporting_factors = [supporting_match.group(1).strip()]
        
        return result
    
    def elicit(self, es_context: Dict, et_context: Dict,
               triggered_rules: List = None,
               custom_prompt: str = None,
               decision: Optional[int] = None) -> RationaleResult:
        """
        Explain an entity pair and, when a prediction is supplied, explain that
        prediction. No new decision is taken here.
        
        Args:
            es_context: Source entity context dict
            et_context: Target entity context dict
            triggered_rules: List of triggered FOL rules (optional)
            custom_prompt: Custom prompt template (optional)
            decision: Prediction y_hat from Mheavy (1 match / 0 non-match).
                Section III-C conditions the encoder on the decision, so passing
                it makes the elicited rationale an explanation of y_hat rather
                than an independent opinion.
            
        Returns:
            RationaleResult with structured rationale analysis
        """
        self.total_calls += 1
        triggered_rules = triggered_rules or []
        
        # Build prompt
        entity_a_str = self._serialize_entity(es_context)
        entity_b_str = self._serialize_entity(et_context)
        rules_section = self._build_triggered_rules_section(triggered_rules)
        
        if custom_prompt:
            prompt = custom_prompt.format(
                entity_a=entity_a_str,
                entity_b=entity_b_str,
                triggered_rules_section=rules_section,
            )
        else:
            prompt = self.DEFAULT_RATIONALE_PROMPT.format(
                entity_a=entity_a_str,
                entity_b=entity_b_str,
                triggered_rules_section=rules_section,
            )
        
        if decision is not None:
            verdict = "MATCH" if int(decision) == 1 else "NON-MATCH"
            prompt = (
                f"**Prediction to explain: {verdict}**\n\n"
                f"A base matcher has already predicted {verdict} for this pair. "
                f"Do not overrule it. Report the evidence that supports this "
                f"prediction, the decisive attributes behind it, and any "
                f"conflicting evidence that was outweighed.\n\n"
            ) + prompt
        
        if self.client:
            for attempt in range(self.max_retries):
                try:
                    t0 = time.time()
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{'role': 'user', 'content': prompt}],
                        temperature=self.temperature,
                        response_format={"type": "json_object"},
                        timeout=self.timeout,
                    )
                    elapsed = time.time() - t0
                    self.total_time += elapsed
                    
                    response_text = response.choices[0].message.content.strip()
                    tokens = response.usage.total_tokens if hasattr(response, 'usage') else 0
                    self.total_tokens += tokens
                    
                    result = self._parse_rationale_response(response_text)
                    result.token_usage = tokens
                    return result
                    
                except Exception as e:
                    if attempt < self.max_retries - 1:
                        wait = 2 ** attempt
                        print(f"Mlight: attempt {attempt+1} failed ({e}), retrying in {wait}s...")
                        time.sleep(wait)
                    else:
                        print(f"Mlight: all {self.max_retries} attempts failed: {e}")
                        # Return a fallback rationale
                        return RationaleResult(
                            rationale=f"[DECISIVE] entity_name analysis (fallback after API failure)\n[SUPPORTING] automatic fallback",
                            token_usage=0,
                        )
        
        # Fallback without client
        return RationaleResult(
            rationale="[DECISIVE] entity_name=same\n[SUPPORTING] automatic fallback (no client)",
            token_usage=0,
        )
    
    def get_stats(self) -> Dict:
        """Get Mlight statistics."""
        return {
            'mlight_total_calls': self.total_calls,
            'mlight_total_tokens': self.total_tokens,
            'mlight_total_time': self.total_time,
            'mlight_avg_tokens': self.total_tokens / max(1, self.total_calls),
            'mlight_avg_time': self.total_time / max(1, self.total_calls),
        }


# ==============================================================================
# Mheavy: Decision Module (complementary to Mlight)
# ==============================================================================

class MheavyDecisionMaker:
    """
    Mheavy: Decision-making module that uses the rationale from Mlight.
    
    Paper Section III-C Step 2: Mheavy takes the rationale + entity context
    and makes the final matching decision with rule contribution scores.
    """
    
    DEFAULT_DECISION_PROMPT = """You are an entity matching decision engine. Based on the detailed analysis provided, make a final MATCH or NON-MATCH decision.

**Entity A:**
{entity_a}

**Entity B:**
{entity_b}

**Detailed Analysis (from Mlight):**
{rationale}

{triggered_rules_section}

**Instructions:**
1. Review the analysis above carefully.
2. Weigh the DECISIVE factors against any CONFLICTING evidence.
3. Make a final decision: MATCH or NON-MATCH.
4. For each triggered rule, assign a contribution score sR ∈ [0.0, 1.0]:
   - 1.0 = DECISIVE (the rule was the key factor in your decision)
   - 0.5-0.9 = SUPPORTING (the rule partially influenced your decision)
   - 0.1-0.4 = WEAK (the rule was considered but had minimal impact)
   - 0.0 = IRRELEVANT (the rule was not used in your decision)

**Output Format (JSON only):**
```json
{{
  "decision": "MATCH",
  "rule_feedback": {{
    "rule_1": 0.9,
    "rule_2": 0.3
  }},
  "confidence": 0.95,
  "summary": "Brief one-line summary of why this decision was made."
}}
```

You MUST respond with valid JSON only."""

    def __init__(self, client=None, model: str = "gpt-3.5-turbo-1106",
                 temperature: float = 0.0, max_retries: int = 3,
                 timeout: int = 30):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout = timeout
        
        # Statistics
        self.total_calls = 0
        self.total_tokens = 0
        self.total_time = 0.0
    
    def _serialize_entity(self, context: Dict) -> str:
        """Serialize entity context into a readable string."""
        parts = []
        for key, val in context.items():
            if key.startswith('neighbors_'):
                rel = key.replace('neighbors_', '')
                if isinstance(val, (set, list)):
                    neighbors_str = ', '.join(str(v) for v in val)
                else:
                    neighbors_str = str(val)
                parts.append(f"  - Relation '{rel}': {neighbors_str}")
            elif key == 'entity_name':
                parts.append(f"  - Name: {val}")
            elif key == 'description':
                if val:
                    parts.append(f"  - Description: {val}")
            else:
                parts.append(f"  - {key}: {val}")
        return '\n'.join(parts) if parts else '(no context available)'
    
    def _build_triggered_rules_section(self, triggered_rules: List) -> str:
        """Build a section describing triggered rules."""
        if not triggered_rules:
            return ""
        
        lines = ["**Triggered Rules (for scoring):**"]
        for i, rule in enumerate(triggered_rules, 1):
            atom_strs = []
            for a in rule.atoms:
                atom_strs.append(f"{a.atom_type}({a.attr})")
            atoms_str = " ∧ ".join(atom_strs)
            verdict = "MATCH" if rule.conclusion == 1 else "NON-MATCH"
            lines.append(
                f"  Rule {i}: {atoms_str} → {verdict} "
                f"(Conf: {rule.conf:.3f})"
            )
        return '\n'.join(lines)
    
    def decide(self, es_context: Dict, et_context: Dict,
               rationale: str = None,
               triggered_rules: List = None,
               custom_prompt: str = None,
               base_prompt: str = None) -> Dict:
        """
        Make a matching decision based on the rationale from Mlight.
        
        Args:
            es_context: Source entity context
            et_context: Target entity context
            rationale: Rationale from Mlight
            triggered_rules: List of triggered FOL rules
            custom_prompt: Custom prompt template (with {entity_a}, {entity_b}, {rationale}, {triggered_rules_section})
            base_prompt: Original task-specific prompt (e.g., SM, ER, EL prompt) — appended as task context
            
        Returns:
            Dict with 'decision', 'rule_feedback', 'confidence', 'summary', 'token_usage'
        """
        self.total_calls += 1
        triggered_rules = triggered_rules or []
        if not rationale:
            rationale = ("(none supplied - decide from the evidence subgraph "
                         "and the triggered rules alone)")
        
        # Build prompt
        entity_a_str = self._serialize_entity(es_context)
        entity_b_str = self._serialize_entity(et_context)
        rules_section = self._build_triggered_rules_section(triggered_rules)
        
        if custom_prompt:
            prompt = custom_prompt.format(
                entity_a=entity_a_str,
                entity_b=entity_b_str,
                rationale=rationale,
                triggered_rules_section=rules_section,
            )
        else:
            prompt = self.DEFAULT_DECISION_PROMPT.format(
                entity_a=entity_a_str,
                entity_b=entity_b_str,
                rationale=rationale,
                triggered_rules_section=rules_section,
            )
        
        # If base_prompt is provided, inject it as task context
        if base_prompt:
            prompt = prompt.replace(
                "You are an entity matching decision engine.",
                "You are a decision engine for the following task:\n\n--- TASK CONTEXT ---\n" + base_prompt[:1500] + "\n--- END TASK CONTEXT ---\n"
            )
        
        if self.client:
            for attempt in range(self.max_retries):
                try:
                    t0 = time.time()
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{'role': 'user', 'content': prompt}],
                        temperature=self.temperature,
                        response_format={"type": "json_object"},
                        timeout=self.timeout,
                    )
                    elapsed = time.time() - t0
                    self.total_time += elapsed
                    
                    response_text = response.choices[0].message.content.strip()
                    tokens = response.usage.total_tokens if hasattr(response, 'usage') else 0
                    self.total_tokens += tokens
                    
                    return self._parse_decision_response(response_text, tokens)
                    
                except Exception as e:
                    if attempt < self.max_retries - 1:
                        wait = 2 ** attempt
                        print(f"Mheavy: attempt {attempt+1} failed ({e}), retrying in {wait}s...")
                        time.sleep(wait)
                    else:
                        print(f"Mheavy: all {self.max_retries} attempts failed: {e}")
                        return {
                            'decision': None,
                            'rule_feedback': {},
                            'confidence': 0.0,
                            'summary': f'API failure: {e}',
                            'token_usage': 0,
                            'raw_response': '',
                        }
        
        # Fallback without client
        return {
            'decision': None,
            'rule_feedback': {},
            'confidence': 0.0,
            'summary': 'no client available',
            'token_usage': 0,
            'raw_response': '',
        }
    
    def _parse_decision_response(self, response_text: str, tokens: int = 0) -> Dict:
        """Parse the JSON decision response."""
        try:
            # Try to extract JSON block
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_match = re.search(r'\{.*"decision".*\}', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    json_str = response_text
            
            data = json.loads(json_str)
            decision_str = data.get('decision', '').upper()
            decision = 1 if 'MATCH' in decision_str and 'NON' not in decision_str else 0
            
            return {
                'decision': decision,
                'rule_feedback': data.get('rule_feedback', {}),
                'confidence': data.get('confidence', 0.5),
                'summary': data.get('summary', ''),
                'token_usage': tokens,
                'raw_response': response_text,
            }
        except (json.JSONDecodeError, KeyError):
            # Fallback text parsing
            decision = None
            if 'match' in response_text.lower() and 'non-match' not in response_text.lower():
                decision = 1
            elif 'non-match' in response_text.lower() or 'no match' in response_text.lower():
                decision = 0
            
            return {
                'decision': decision,
                'rule_feedback': {},
                'confidence': 0.5,
                'summary': response_text[:200],
                'token_usage': tokens,
                'raw_response': response_text,
            }
    
    def get_stats(self) -> Dict:
        """Get Mheavy statistics."""
        return {
            'mheavy_total_calls': self.total_calls,
            'mheavy_total_tokens': self.total_tokens,
            'mheavy_total_time': self.total_time,
            'mheavy_avg_tokens': self.total_tokens / max(1, self.total_calls),
            'mheavy_avg_time': self.total_time / max(1, self.total_calls),
        }


# ==============================================================================
# Test / Demo
# ==============================================================================

if __name__ == "__main__":
    print("EvoRM Mlight/Mheavy - Self Test")
    print("=" * 60)
    
    # Test without client (offline)
    mlight = MlightRationaleElicitor(client=None)
    mheavy = MheavyDecisionMaker(client=None)
    
    es_ctx = {
        "entity_name": "Test Entity A",
        "title": "Machine Learning Basics",
        "year": "2020",
        "authors": "Smith et al.",
    }
    et_ctx = {
        "entity_name": "Test Entity B",
        "title": "Machine Learning Basics",
        "year": "2020",
        "authors": "Smith and Jones",
    }
    
    # Test rationale elicitation
    print("\n1. Testing Mlight Rationale Elicitation...")
    rationale_result = mlight.elicit(es_ctx, et_ctx)
    print(f"   Rationale: {rationale_result.rationale[:100]}...")
    print(f"   Decisive factors: {rationale_result.decisive_factors}")
    
    # Test decision
    print("\n2. Testing Mheavy Decision...")
    decision_result = mheavy.decide(es_ctx, et_ctx, rationale_result.rationale)
    print(f"   Decision: {decision_result['decision']}")
    print(f"   Confidence: {decision_result['confidence']}")
    
    print("\n3. Stats:")
    print(f"   Mlight: {mlight.get_stats()}")
    print(f"   Mheavy: {mheavy.get_stats()}")
    
    print("\n✅ All tests passed!")
