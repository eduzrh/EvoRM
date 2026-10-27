# Plugging EvoRM into a baseline

EvoRM is model-agnostic. The baselines compared against in the paper keep their
own repositories and evaluation pipelines; use the port in `evorm/port.py` to
attach EvoRM to whichever framework you are running.

## Entity resolution — MatchGPT / AnyMatch / Ditto

These frameworks emit a binary decision per record pair.

```python
from evorm.port import EvoRMPort

def matcher(request):
    answer = baseline.match(request["es_context"], request["et_context"])
    return {"decision": int(answer.label),
            "rationale": answer.rationale,
            "token_usage": answer.tokens}

port = EvoRMPort(base_matcher=matcher)
label = port.decide(a["id"], b["id"], a, b).label
```

## Entity alignment — ChatEA / AdacoAgentEA

These pipelines retrieve candidates and rank them. Two options:

```python
# (a) rerank only
ranked = port.rerank(qid, q_ctx,
                     [(c.id, c) for c in candidates],
                     base_scores={c.id: c.score for c in candidates})

# (b) learn from the pipeline's own trajectories, then rerank
for traj in pipeline.trajectories:      # decision + rationale
    port.observe(traj.a_id, traj.b_id, traj.a, traj.b,
                 decision=traj.decision, rationale=traj.rationale)
```

## Entity linking — LELA

LELA produces a decision plus a rationale for each mention/candidate pair.

```python
for mention in cold_start_mentions:
    result = lela.link(mention)
    port.observe(mention.id, result.entity_id,
                 mention.context, result.entity_context,
                 decision=1, rationale=result.rationale)
```

Then rescore the top-64 BM25 candidates with `port.rerank` and compute
normalised accuracy with `evorm.metrics.normalized_accuracy`, aggregating the
four ZESHEL domains with `macro_average`.

## Schema matching — LLM-DP / ReMatch / Matchmaker

Binary decision per column pair; identical to the ER pattern. Accuracy@1 equals
F1 in the many-to-one setup.
