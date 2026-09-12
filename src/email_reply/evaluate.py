"""Evaluation harness: automated metrics + LLM-as-judge + baselines comparison.

Metrics:
  1. Intent accuracy (vs golden labels)
  2. Reply quality (LLM-as-judge: relevance, helpfulness, brand-voice, factual accuracy)
  3. Escalation precision/recall (vs golden labels)
  4. Automated text metrics (ROUGE-L-ish, length appropriateness)
"""
from __future__ import annotations

import asyncio
import json
import re
import random
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from email_reply.llm import LLM, extract_json

DATA = Path(__file__).resolve().parents[2] / "data"
REPORTS = Path(__file__).resolve().parents[2] / "reports"

# ============================================================
# LLM-AS-JUDGE
# ============================================================

JUDGE_PROMPT = """You are evaluating a customer-support reply from @AppleSupport on Twitter.

Customer message: "{customer}"
Brand's actual historical reply: "{reference}"
Generated reply to evaluate: "{generated}"

Rate the generated reply on these dimensions (1-5 each):

1. RELEVANCE: Does it address the specific issue the customer raised? (1=completely off-topic, 5=directly addresses the exact problem)
2. HELPFULNESS: Does it provide actionable next steps or useful information? (1=no value, 5=clear actionable solution)
3. BRAND_VOICE: Does it sound like @AppleSupport? (concise, professional, empathetic, technical but accessible) (1=wrong tone, 5=perfect Apple voice)
4. ACCURACY: Are the technical suggestions/info factually correct for Apple products? (1=wrong info, 5=technically sound)
5. ESCALATION_APPROPRIATENESS: For complex issues, does it appropriately suggest DM/call/store visit? For simple ones, does it handle directly? (1=badly misjudged, 5=perfect triage)

Also provide:
- overall: weighted average (relevance 0.3, helpfulness 0.25, accuracy 0.2, brand_voice 0.15, escalation 0.1)
- critique: one sentence on the biggest weakness
- would_send: true/false — would you actually send this to a real customer?

Return JSON:
{{
  "relevance": <1-5>,
  "helpfulness": <1-5>,
  "accuracy": <1-5>,
  "brand_voice": <1-5>,
  "escalation_appropriateness": <1-5>,
  "overall": <1.0-5.0>,
  "critique": "...",
  "would_send": true/false
}}"""


@dataclass
class JudgeScore:
    relevance: float
    helpfulness: float
    accuracy: float
    brand_voice: float
    escalation_appropriateness: float
    overall: float
    critique: str
    would_send: bool


async def judge_reply(llm: LLM, customer: str, reference: str, generated: str) -> JudgeScore:
    prompt = JUDGE_PROMPT.format(
        customer=customer[:400], reference=reference[:400], generated=generated[:400])
    raw = await llm.generate(prompt, temperature=0.0, json_mode=True)
    d = extract_json(raw)
    return JudgeScore(
        relevance=float(d.get("relevance", 3)),
        helpfulness=float(d.get("helpfulness", 3)),
        accuracy=float(d.get("accuracy", 3)),
        brand_voice=float(d.get("brand_voice", 3)),
        escalation_appropriateness=float(d.get("escalation_appropriateness", 3)),
        overall=float(d.get("overall", 3.0)),
        critique=d.get("critique", ""),
        would_send=bool(d.get("would_send", False)),
    )


# ============================================================
# AUTOMATED TEXT METRICS
# ============================================================

def rouge_l(reference: str, candidate: str) -> float:
    """Simple ROUGE-L (longest common subsequence) F1."""
    ref_toks = reference.lower().split()
    cand_toks = candidate.lower().split()
    if not ref_toks or not cand_toks:
        return 0.0
    m, n = len(ref_toks), len(cand_toks)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_toks[i-1] == cand_toks[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    p = lcs / n if n else 0
    r = lcs / m if m else 0
    return 2 * p * r / (p + r) if (p + r) else 0.0


def length_appropriateness(text: str) -> float:
    """Score 0-1: Twitter replies should be concise (< 280 chars ideal, < 560 ok)."""
    n = len(text)
    if n <= 280:
        return 1.0
    elif n <= 560:
        return 0.7
    elif n <= 800:
        return 0.4
    return 0.2


def keyword_overlap(customer: str, reply: str) -> float:
    """Fraction of customer's key terms addressed in the reply."""
    stop = {'i','my','the','a','an','to','is','it','and','of','for','in','on','me','you','your',
            'this','that','have','has','been','was','with','not','can','but','do','just','from'}
    c_words = {w for w in re.findall(r'[a-z]+', customer.lower()) if w not in stop and len(w) > 2}
    r_words = set(re.findall(r'[a-z]+', reply.lower()))
    if not c_words:
        return 1.0
    return len(c_words & r_words) / len(c_words)


# ============================================================
# INTENT ACCURACY
# ============================================================

def intent_metrics(predictions: list[str], labels: list[str]) -> dict:
    """Compute accuracy, per-class precision/recall/F1."""
    assert len(predictions) == len(labels)
    n = len(predictions)
    correct = sum(1 for p, l in zip(predictions, labels) if p == l)
    accuracy = correct / n if n else 0
    
    classes = sorted(set(labels) | set(predictions))
    per_class = {}
    for c in classes:
        tp = sum(1 for p, l in zip(predictions, labels) if p == c and l == c)
        fp = sum(1 for p, l in zip(predictions, labels) if p == c and l != c)
        fn = sum(1 for p, l in zip(predictions, labels) if p != c and l == c)
        prec = tp / (tp + fp) if (tp + fp) else 0
        rec = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        per_class[c] = {"precision": round(prec, 3), "recall": round(rec, 3),
                        "f1": round(f1, 3), "support": tp + fn}
    
    return {"accuracy": round(accuracy, 4), "n": n, "per_class": per_class}


# ============================================================
# ESCALATION METRICS
# ============================================================

def escalation_metrics(predictions: list[bool], labels: list[bool]) -> dict:
    tp = sum(1 for p, l in zip(predictions, labels) if p and l)
    fp = sum(1 for p, l in zip(predictions, labels) if p and not l)
    fn = sum(1 for p, l in zip(predictions, labels) if not p and l)
    tn = sum(1 for p, l in zip(predictions, labels) if not p and not l)
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    return {
        "precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy": round((tp + tn) / len(predictions), 3) if predictions else 0,
    }


# ============================================================
# FULL EVALUATION RUNNER
# ============================================================

@dataclass
class EvalResult:
    id: int
    customer_text: str
    reference_reply: str
    generated_reply: str
    intent_pred: str
    intent_label: str | None
    escalate_pred: bool
    escalate_label: bool | None
    judge_score: JudgeScore | None
    rouge_l: float
    length_score: float
    keyword_overlap: float
    mode: str


async def run_evaluation(
    golden: list[dict],
    agent,  # AppleSupportAgent
    judge_llm: LLM | None = None,
    mode: str = "llm",
    max_n: int | None = None,
) -> dict:
    """Run full evaluation on golden set."""
    if max_n:
        golden = golden[:max_n]
    
    results: list[EvalResult] = []
    
    for i, g in enumerate(golden):
        print(f"  [{i+1}/{len(golden)}] ", end="", flush=True)
        
        # Run agent
        resp = await agent.process(g["customer_text"], mode=mode)
        
        # Automated metrics
        rl = rouge_l(g["brand_text"], resp.reply)
        ls = length_appropriateness(resp.reply)
        ko = keyword_overlap(g["customer_text"], resp.reply)
        
        # LLM judge
        js = None
        if judge_llm:
            try:
                js = await judge_reply(judge_llm, g["customer_text"],
                                       g["brand_text"], resp.reply)
            except Exception as e:
                print(f"judge error: {e}", end=" ")
        
        er = EvalResult(
            id=i, customer_text=g["customer_text"],
            reference_reply=g["brand_text"], generated_reply=resp.reply,
            intent_pred=resp.intent, intent_label=g.get("intent_label"),
            escalate_pred=resp.escalate, escalate_label=g.get("escalate_label"),
            judge_score=js, rouge_l=rl, length_score=ls,
            keyword_overlap=ko, mode=mode,
        )
        results.append(er)
        
        j_str = f"judge={js.overall:.1f}" if js else "no-judge"
        print(f"{resp.intent:20s} esc={'Y' if resp.escalate else 'N'} "
              f"rouge={rl:.2f} {j_str}")
    
    # Aggregate
    report = aggregate_results(results, mode)
    return report


def aggregate_results(results: list[EvalResult], mode: str) -> dict:
    """Aggregate per-response results into overall metrics."""
    n = len(results)
    
    # Intent metrics (where labels exist)
    labeled = [(r.intent_pred, r.intent_label) for r in results if r.intent_label]
    intent_report = {}
    if labeled:
        preds, labels = zip(*labeled)
        intent_report = intent_metrics(list(preds), list(labels))
    
    # Escalation metrics (where labels exist)
    esc_labeled = [(r.escalate_pred, r.escalate_label) for r in results if r.escalate_label is not None]
    esc_report = {}
    if esc_labeled:
        preds, labels = zip(*esc_labeled)
        esc_report = escalation_metrics(list(preds), list(labels))
    
    # Judge scores
    judged = [r for r in results if r.judge_score]
    judge_report = {}
    if judged:
        judge_report = {
            "mean_relevance": round(sum(r.judge_score.relevance for r in judged) / len(judged), 3),
            "mean_helpfulness": round(sum(r.judge_score.helpfulness for r in judged) / len(judged), 3),
            "mean_accuracy": round(sum(r.judge_score.accuracy for r in judged) / len(judged), 3),
            "mean_brand_voice": round(sum(r.judge_score.brand_voice for r in judged) / len(judged), 3),
            "mean_escalation_app": round(sum(r.judge_score.escalation_appropriateness for r in judged) / len(judged), 3),
            "mean_overall": round(sum(r.judge_score.overall for r in judged) / len(judged), 3),
            "would_send_rate": round(sum(1 for r in judged if r.judge_score.would_send) / len(judged), 3),
            "n_judged": len(judged),
        }
    
    # Automated metrics
    auto_report = {
        "mean_rouge_l": round(sum(r.rouge_l for r in results) / n, 4) if n else 0,
        "mean_length_score": round(sum(r.length_score for r in results) / n, 4) if n else 0,
        "mean_keyword_overlap": round(sum(r.keyword_overlap for r in results) / n, 4) if n else 0,
    }
    
    # Per-response detail
    per_response = []
    for r in results:
        pr = {
            "id": r.id,
            "customer_text": r.customer_text[:200],
            "generated_reply": r.generated_reply[:300],
            "reference_reply": r.reference_reply[:300],
            "intent_pred": r.intent_pred,
            "intent_label": r.intent_label,
            "escalate_pred": r.escalate_pred,
            "escalate_label": r.escalate_label,
            "rouge_l": round(r.rouge_l, 4),
            "mode": r.mode,
        }
        if r.judge_score:
            pr["judge"] = {
                "overall": r.judge_score.overall,
                "relevance": r.judge_score.relevance,
                "helpfulness": r.judge_score.helpfulness,
                "accuracy": r.judge_score.accuracy,
                "brand_voice": r.judge_score.brand_voice,
                "critique": r.judge_score.critique,
                "would_send": r.judge_score.would_send,
            }
        per_response.append(pr)
    
    return {
        "mode": mode,
        "n": n,
        "intent": intent_report,
        "escalation": esc_report,
        "judge": judge_report,
        "automated": auto_report,
        "per_response": per_response,
    }
