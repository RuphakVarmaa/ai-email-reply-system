"""Metric validation: does the metric suite actually measure reply quality?

Three independent validations (the core of the eval section):

V1. PERTURBATION SENSITIVITY — take known-good replies (the dataset's own
    reference replies), corrupt each in one specific way, and require the
    metric to punish it materially:
      P1 wrong refund window  (e.g. "5–7 business days" -> "3–4 business days")
      P2 wrong order number   (swap digits)
      P3 tone inversion       (prepend a cold, defensive opener)
      P4 coverage hole        (delete the paragraph answering question 2)
      P5 forbidden phrase     (insert "As an AI, I cannot...")
      P6 wrong decision       (in a warranty-void scenario, promise a replacement)
    Expected: corrupt replies score clearly below originals (drop > 0.10 in
    mean composite), and specifically the corrupted dimension drops the most
    (e.g. P1 lowers faithfulness/contradiction-related checks the most).

V2. HUMAN-LABEL CORRELATION — 50 (reply, human quality 1-5) pairs, hand-labeled
    under a documented rubric (see data/calibration/rubric.md), blind to metric
    output. Report Spearman rho between composite and human labels, and per-
    metric rho. Also report the alternative-composite correlations to justify
    the weight choice (coverage-heavy vs similarity-heavy etc.).

V3. INTER-JUDGE AGREEMENT — the same replies judged by a second judge model
    (different family). Report Pearson/Spearman between judge overall scores
    and between composite rankings. High agreement + high human correlation
    = the judge is not just noise.
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

DATA = Path(__file__).resolve().parents[3] / "data"


# ------------------------------------------------------------------ V1 perturbations

@dataclass
class Perturbation:
    id: str
    name: str
    fn: callable  # type: ignore[valid-type]
    expected_signal: str  # which dimension should drop


def p1_wrong_refund_window(reply: str, facts: dict, kb: dict) -> str:
    """Change the refund-window claim to a wrong one."""
    text = kb["policies"]["returns"]["refund_window_text"]
    m = re.search(r"(\d+)\s*[–—-]\s*(\d+)\s*(?:business\s*)?days", reply)
    if not m:
        # insert a wrong window after a refund mention
        if re.search(r"refund", reply, re.I):
            return re.sub(r"(refund[^.]{0,60})\.", r"\1 within 3–4 business days.",
                          reply, count=1, flags=re.I)
        return reply + " Your refund will arrive within 3–4 business days."
    wrong = f"{max(int(m.group(1)) - 3, 1)}–{max(int(m.group(2)) - 3, 2)} business days"
    return reply[:m.start()] + wrong + reply[m.end():]


def p2_wrong_order(reply: str, facts: dict, kb: dict) -> str:
    """Swap digits in the order number."""
    order = facts.get("order_number")
    if not order:
        return reply
    digits = list(order)
    for i in range(len(digits) - 1, -1, -1):
        if digits[i].isdigit() and digits[i] != "9":
            digits[i] = str(int(digits[i]) + 1)
            break
    wrong = "".join(digits)
    return reply.replace(order, wrong)


def p3_tone_inversion(reply: str, facts: dict, kb: dict) -> str:
    """Prepend a cold, defensive, bureaucratic opener."""
    return ("Per our company policy, all requests are processed strictly in the "
            "order received. We are not able to make exceptions and we do not "
            "offer compensations of any kind. " + reply)


def p4_coverage_hole(reply: str, facts: dict, kb: dict) -> str:
    """Delete the sentence most likely answering a secondary question."""
    sentences = re.split(r"(?<=[.!?])\s+", reply)
    if len(sentences) <= 2:
        return reply.replace("\n", " ")  # degenerate: return as-is (marked)
    # remove the 2nd sentence block (usually the second question's answer)
    del sentences[min(2, len(sentences) - 1)]
    return " ".join(sentences)


def p5_forbidden(reply: str, facts: dict, kb: dict) -> str:
    return reply + "\n\nAs an AI, I cannot guarantee these timelines."


def p6_wrong_decision(reply: str, facts: dict, kb: dict) -> str:
    """Promise the wrong action."""
    if facts.get("in_warranty") is False:
        return reply + " Actually, we'll make an exception and ship you a free replacement immediately."
    if facts.get("adjustment_approved") is False:
        return reply + " We've gone ahead and refunded the price difference to your card."
    if facts.get("already_shipped"):
        return reply + " Your order has been cancelled and no further charges will occur."
    return reply + " We've issued you a full refund plus a 50% goodwill discount."


PERTURBATIONS = [
    Perturbation("P1", "wrong_refund_window", p1_wrong_refund_window, "faithfulness"),
    Perturbation("P2", "wrong_order_number", p2_wrong_order, "entity_accuracy"),
    Perturbation("P3", "tone_inversion", p3_tone_inversion, "tone"),
    Perturbation("P4", "coverage_hole", p4_coverage_hole, "coverage"),
    Perturbation("P5", "forbidden_phrase", p5_forbidden, "gates"),
    Perturbation("P6", "wrong_decision", p6_wrong_decision, "action/faithfulness"),
]


def build_perturbation_set(test_pairs: list[dict], n: int = 24, seed: int = 3):
    """Select n test pairs; produce (pair, original, corrupted, perturbation)."""
    rng = random.Random(seed)
    kb = json.loads("{}")
    import yaml
    kb = yaml.safe_load((DATA / "knowledge_base.yaml").read_text())
    chosen = rng.sample(test_pairs, min(n, len(test_pairs)))
    cases = []
    for p in chosen:
        pert = rng.choice(PERTURBATIONS)
        corrupted = pert.fn(p["reply"], p["facts"], kb)
        cases.append({
            "id": p["id"], "intent": p["intent"],
            "original_reply": p["reply"], "corrupted_reply": corrupted,
            "perturbation": pert.id + ":" + pert.name,
            "expected_signal": pert.expected_signal,
        })
    return cases


# ------------------------------------------------------------------ V2/V3 stats

def spearman(x: list[float], y: list[float]) -> float:
    """Spearman rank correlation (pure numpy/numpy-free implementation)."""
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = ranks(x), ranks(y)
    return pearson(rx, ry)


def pearson(x: list[float], y: list[float]) -> float:
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den = (sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y)) ** 0.5
    return num / den if den else 0.0
