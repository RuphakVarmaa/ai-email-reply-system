"""Composite scoring: per-response score + evidence + overall system score.

Per-response composite:
  S = 0.30·coverage + 0.25·faithfulness + 0.20·action + 0.15·tone_norm
      + 0.10·entity_accuracy
subject to hard gates:
  G_a faithfulness ≥ 0.75
  G_b contradiction_rate ≤ 0.10
  G_c zero major-gate failures (forbidden phrase, prompt leak, wrong decision)
Gate failure multiplies the score by 0.5 (per failed gate family), never zero,
so that gradients survive for analysis — but a reply failing the decision gates
can never exceed 0.5 overall, which is the honest signal that a human must
rewrite it.

Also computed: embedding similarity to the reference reply (reported, low
weight in a 'similarity' column, NOT part of the composite by default — see
README for why surface/semantic overlap is a weak target).

Overall system score:
  - macro mean of per-response composites
  - bootstrap 95% CI over test set
  - per-intent and per-register breakdown
  - gate failure rates
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from .facts import FactReport, run_fact_checks

WEIGHTS = {
    "coverage": 0.30,
    "faithfulness": 0.25,
    "action_correctness": 0.20,
    "tone": 0.15,
    "entity_accuracy": 0.10,
}

GATES = {
    "faithfulness_min": 0.75,
    "contradiction_max": 0.10,
}


@dataclass
class ResponseScore:
    id: str
    intent: str
    register: str
    composite: float
    components: dict
    gate_failures: list
    fact_checks: list
    judge_coverage_detail: list
    claims: list
    verdicts: list
    similarity: float | None = None
    reply: str = ""


def tone_norm(tone_1_5: float) -> float:
    """Map 1..5 to 0..1 with 4->0.875, 5->1."""
    return (tone_1_5 - 1) / 4.0


def score_response(judge_result, facts: dict, kb: dict, reply: str) -> ResponseScore:
    """Combine programmatic fact checks + judge outputs into a ResponseScore."""
    fr: FactReport = run_fact_checks(reply, facts, kb)
    entity_acc = fr.entity_accuracy
    if entity_acc is None:
        entity_acc = 0.5  # no applicable entity checks — neutral

    components = {
        "coverage": judge_result.coverage,
        "faithfulness": judge_result.faithfulness,
        "action_correctness": judge_result.action_correct,
        "tone": tone_norm(judge_result.tone),
        "entity_accuracy": entity_acc,
    }

    gate_failures = []
    if judge_result.faithfulness < GATES["faithfulness_min"]:
        gate_failures.append(f"faithfulness {judge_result.faithfulness:.2f} < {GATES['faithfulness_min']}")
    if judge_result.contradiction_rate > GATES["contradiction_max"]:
        gate_failures.append(f"contradiction {judge_result.contradiction_rate:.2f} > {GATES['contradiction_max']}")
    majors = [c for c in fr.checks if c.verdict == "FAIL" and c.severity == "major"]
    if majors:
        gate_failures.append(f"{len(majors)} major fact-check failure(s): "
                             + "; ".join(c.detail for c in majors[:3]))

    raw = sum(WEIGHTS[k] * components[k] for k in WEIGHTS)
    penalty = 0.5 ** min(len(gate_failures), 2) if gate_failures else 1.0
    composite = raw * penalty

    return ResponseScore(
        id="", intent="", register="",
        composite=round(composite, 4),
        components={k: round(v, 4) for k, v in components.items()},
        gate_failures=gate_failures,
        fact_checks=[{"name": c.name, "verdict": c.verdict,
                      "detail": c.detail, "severity": c.severity} for c in fr.checks],
        judge_coverage_detail=judge_result.coverage_detail,
        claims=judge_result.claims,
        verdicts=judge_result.verdicts,
        reply=reply,
    )


def overall_scores(scores: list[ResponseScore]) -> dict:
    """Aggregate: mean composite, bootstrap CI, per-intent/register, gate rates."""
    if not scores:
        return {"n": 0}
    comps = np.array([s.composite for s in scores])
    rng = random.Random(0)
    n = len(scores)
    boots = []
    for _ in range(2000):
        idx = [rng.randrange(n) for _ in range(n)]
        boots.append(np.mean(comps[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])

    by_intent = {}
    for s in scores:
        by_intent.setdefault(s.intent, []).append(s.composite)
    by_register = {}
    for s in scores:
        by_register.setdefault(s.register, []).append(s.composite)

    n_gate_fail = sum(1 for s in scores if s.gate_failures)

    return {
        "n": n,
        "mean_composite": round(float(np.mean(comps)), 4),
        "ci95": [round(float(lo), 4), round(float(hi), 4)],
        "median_composite": round(float(np.median(comps)), 4),
        "mean_components": {
            k: round(float(np.mean([s.components[k] for s in scores])), 4)
            for k in WEIGHTS
        },
        "gate_failure_rate": round(n_gate_fail / n, 4),
        "by_intent": {k: round(float(np.mean(v)), 4) for k, v in sorted(by_intent.items())},
        "by_register": {k: round(float(np.mean(v)), 4) for k, v in sorted(by_register.items())},
        "score_histogram": np.histogram(comps, bins=[0, .2, .4, .6, .8, 1.0])[0].tolist(),
        "n_below_050": int((comps < 0.5).sum()),
        "n_above_080": int((comps >= 0.8).sum()),
    }
