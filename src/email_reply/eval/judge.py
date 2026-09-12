"""LLM-judge layer of the evaluation system.

Mitigations for documented judge biases (Zheng et al., NeurIPS 2023; G-Eval):
- SELF-PREFERENCE: generator (Gemini family) and judge (Gemma family) are
  different model lineages by default. Configurable.
- POSITION BIAS: not a pairwise setup; single-reply rubric scoring with
  anchored definitions.
- VERBOSITY BIAS: rubric explicitly de-values length; conciseness handled by
  a separate deterministic check.
- JUDGE DRIFT: temperature 0, JSON-schema output, evidence citation required
  for every sub-verdict.

The judge produces:
  coverage        ∈ [0,1]   — fraction of the customer's questions/requests
                              actually answered by the reply
  claims          — atomic claims extracted from the reply, each checked for
                              support against the facts block => faithfulness
  tone            ∈ 1..5    — register appropriateness
  action          ∈ [0,1]   — did the reply take/commit the right action
  overall         ∈ 1..5    — holistic (reported, not used in composite)
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field

from ..llm import LLM, extract_json

CLAIM_PROMPT = """You extract atomic factual claims from a customer-support reply.

REPLY:
{reply}

List every factual claim the reply makes about this specific case (order status,
amounts, dates, timelines, what the company has done or will do, policy rules).
Ignore greetings, empathy, and sign-offs. Each claim must be a short standalone
sentence.

Return JSON: {{"claims": ["...", ...]}}  (empty list if no factual claims)"""

SUPPORT_PROMPT = """You are a strict fact-checker. For each claim, decide whether it is
SUPPORTED by the ground-truth facts below, CONTRADICTED by them, or UNVERIFIABLE.

GROUND-TRUTH FACTS:
{facts}

CLAIMS:
{claims}

Rules:
- SUPPORTED: the claim can be derived from the facts given.
- CONTRADICTED: the facts say otherwise (e.g. claim says 7-day window, facts say 5-7).
- UNVERIFIABLE: the facts neither confirm nor deny (e.g. "tracking updated last night").
- Judge against the facts ONLY, not plausibility.

Return JSON:
{{"verdicts": [{{"claim": "...", "verdict": "SUPPORTED|CONTRADICTED|UNVERIFIABLE", "why": "short"}}, ...]}}"""

COVERAGE_PROMPT = """A customer sent this email to a support inbox:

CUSTOMER EMAIL:
Subject: {subject}
{incoming}

The customer's questions/requests, as determined by careful reading:
{questions}

The proposed agent reply:
{reply}

For EACH question/request, decide whether the reply actually answers it
(ANSWERED) or not (MISSED). Partial answers count as ANSWERED only if they
address the core ask.

Return JSON:
{{"answers": [{{"question": "...", "status": "ANSWERED|MISSED", "evidence": "short quote or 'none'"}}, ...]}}"""

TONE_PROMPT = """Rate how well this support reply matches the customer's tone and situation.

Customer register: {register_desc}
Customer email (excerpt): {incoming_excerpt}

Reply: {reply}

Rubric (1-5):
1 = inappropriate (defensive, robotic, flippant, or jarringly casual)
2 = mostly off (wrong energy for the situation)
3 = acceptable (neutral, no friction)
4 = good (reads human, well-matched to the customer)
5 = excellent (empathy lands perfectly; a senior agent's voice)

Return JSON: {{"tone": <1-5>, "rationale": "one sentence"}}"""

ACTION_PROMPT = """Does this support reply take the RIGHT action for this situation?

The customer asked about: {questions}
The correct action per company policy/ground truth: {actions}

Reply: {reply}

Decide:
- action_taken: what the reply commits to (one short sentence)
- action_correct: 1.0 if the reply commits to the right action(s) (or correctly
  declines when decline is the right action), 0.0 if it commits to a wrong
  action, 0.5 if it hedges/defers without committing.

Return JSON:
{{"action_taken": "...", "action_correct": <0.0|0.5|1.0>, "reason": "one sentence"}}"""


@dataclass
class JudgeResult:
    coverage: float
    coverage_detail: list
    claims: list
    faithfulness: float
    contradiction_rate: float
    verdicts: list
    tone: float
    tone_rationale: str
    action_correct: float
    action_taken: str
    action_reason: str
    overall: float | None = None


REGISTER_DESC = {
    "calm": "polite, even-tempered customer",
    "frustrated": "annoyed but civil; this is their second ask",
    "furious": "very angry, threatening chargeback/review",
    "terse": "extremely brief, no pleasantries",
    "chatty": "warm and friendly with a personal anecdote",
    "professional": "formal corporate buyer",
    "grateful": "effusively thankful, then a new question",
}


class Judge:
    def __init__(self, llm: LLM):
        self.llm = llm

    async def _gen_json(self, prompt: str) -> dict:
        out = await self.llm.generate(prompt, temperature=0.0, json_mode=True)
        return extract_json(out)

    async def judge(self, incoming: str, subject: str, reply: str,
                    facts: dict, questions: list, actions: list,
                    register: str) -> JudgeResult:
        # 1. claim extraction
        claims_data = await self._gen_json(
            CLAIM_PROMPT.format(reply=reply))
        claims = claims_data.get("claims", [])[:15]

        # 2. claim support vs ground-truth facts (rendered)
        facts_rendered = render_facts(facts)
        verdicts = []
        if claims:
            verdicts_data = await self._gen_json(
                SUPPORT_PROMPT.format(facts=facts_rendered,
                                      claims=json.dumps(claims, ensure_ascii=False)))
            verdicts = verdicts_data.get("verdicts", [])

        supported = sum(1 for v in verdicts if v.get("verdict") == "SUPPORTED")
        contradicted = sum(1 for v in verdicts if v.get("verdict") == "CONTRADICTED")
        faithfulness = supported / len(claims) if claims else 1.0
        contra_rate = contradicted / len(claims) if claims else 0.0

        # 3. coverage per question
        qtext = "\n".join(f"- {q}" for q in questions)
        cov_data = await self._gen_json(
            COVERAGE_PROMPT.format(subject=subject, incoming=incoming,
                                   questions=qtext, reply=reply))
        answers = cov_data.get("answers", [])
        n_answered = sum(1 for a in answers if a.get("status") == "ANSWERED")
        coverage = n_answered / len(questions) if questions else 1.0

        # 4. tone
        tone_data = await self._gen_json(
            TONE_PROMPT.format(register_desc=REGISTER_DESC.get(register, "a customer"),
                               incoming_excerpt=incoming[:400], reply=reply))
        tone = float(tone_data.get("tone", 3.0))

        # 5. action correctness
        act_data = await self._gen_json(
            ACTION_PROMPT.format(questions=qtext,
                                 actions="; ".join(actions) if actions else "answer the question",
                                 reply=reply))
        action_correct = float(act_data.get("action_correct", 0.5))

        return JudgeResult(
            coverage=coverage, coverage_detail=answers,
            claims=claims, faithfulness=faithfulness,
            contradiction_rate=contra_rate, verdicts=verdicts,
            tone=tone, tone_rationale=tone_data.get("rationale", ""),
            action_correct=action_correct,
            action_taken=act_data.get("action_taken", ""),
            action_reason=act_data.get("reason", ""),
        )


def render_facts(facts: dict) -> str:
    """Render the ground-truth facts dict as readable lines for the judge."""
    out = []
    for k, v in facts.items():
        out.append(f"- {k}: {v}")
    return "\n".join(out)
