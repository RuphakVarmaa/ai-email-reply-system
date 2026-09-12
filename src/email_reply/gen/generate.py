"""Suggested-reply generator: RAG over past pairs + KB grounding.

Modes (for the ablation study):
  rag       — retrieval few-shots + KB facts (the full system)
  zero_shot — no retrieval, no KB (LLM alone)
  no_kb     — retrieval few-shots, but KB facts withheld
  nn        — pure nearest-neighbor: return the retrieved pair's reply verbatim
              (classical-retrieval baseline, no LLM)

Design choices justified in README §Generator:
- Few-shot via retrieval beats fine-tuning at this corpus size (350 pairs) and
  keeps facts injectable at inference (prices/policies change).
- KB facts are injected explicitly so the model cannot hallucinate them.
- Order/identity facts from the incoming email are forwarded verbatim.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ..llm import LLM
from .retrieve import HybridRetriever, Retrieved, load_pairs

DATA = Path(__file__).resolve().parents[3] / "data"


def load_kb() -> dict:
    with open(DATA / "knowledge_base.yaml") as f:
        return yaml.safe_load(f)


def kb_facts_for(retrieved: list[Retrieved], kb: dict, intent: str | None = None) -> str:
    """Render the KB lines relevant to the anchors cited by retrieved pairs."""
    anchors = set()
    for r in retrieved:
        anchors.update(r.kb_anchors)
    lines = []
    pol = kb["policies"]
    if any(a.startswith("policies.returns") for a in anchors):
        lines.append(f"- Returns: {pol['returns']['window_days']}-day window from delivery; "
                     f"condition: {pol['returns']['condition']}; no restocking fee; "
                     f"refund to original payment method, appears in {pol['returns']['refund_window_text']}; "
                     f"return shipping: {pol['returns']['return_shipping']}; "
                     f"exchanges for size/color: {pol['returns']['exchange_shipping']}.")
    if any(a.startswith("policies.shipping") for a in anchors):
        s = pol["shipping"]
        lines.append(f"- Shipping: standard {s['domestic_standard_days'][0]}–{s['domestic_standard_days'][1]} business days; "
                     f"express {s['domestic_express_days'][0]}–{s['domestic_express_days'][1]} business days for "
                     f"${s['express_upgrade_fee_usd']}; free standard shipping over ${s['free_shipping_threshold_usd']}; "
                     f"orders before {s['cutoff_time']} ship same day; {s['tracking_note']}.")
    if any(a.startswith("policies.warranty") for a in anchors):
        w = pol["warranty"]
        lines.append(f"- Warranty: {w['duration_months']} months; covers {', '.join(w['covers'])}; "
                     f"NOT covered: {', '.join(w['not_covered'])}; process: {w['process']}.")
    if any(a.startswith("policies.orders") for a in anchors):
        o = pol["orders"]
        lines.append(f"- Orders: changes within {o['modification_window_hours']}h; "
                     f"cancellation before ship: {'yes' if o['cancellation_before_ship'] else 'no'}; "
                     f"price adjustment within {o['price_adjustment_days']} days; "
                     f"address changes: {o['address_change']}.")
    if any(a.startswith("policies.gift_cards") for a in anchors):
        g = pol["gift_cards"]
        lines.append(f"- Gift cards: valid {g['expiry_months']} months; combinable with discounts: "
                     f"{'yes' if g['combinable_with_discounts'] else 'no'}; "
                     f"physical cards: {g['physical_card_shipping']}.")
    # product anchors
    for a in anchors:
        if a.startswith("catalog."):
            sku = a.split(".", 1)[1]
            for p in kb["catalog"]:
                if p["sku"] == sku:
                    extra = []
                    if p.get("sizes"):
                        extra.append(f"sizes {', '.join(map(str, p['sizes']))}")
                    if p.get("colors"):
                        extra.append(f"colors {', '.join(p['colors'])}")
                    lines.append(f"- Product {p['name']} ({sku}): ${p['price_usd']:.2f}"
                                  + (f"; {'; '.join(extra)}" if extra else "")
                                  + f"; stock: {p['stock_status']}.")
    return "\n".join(lines)


SYSTEM_PROMPT = """You are a senior customer-support agent at Northwind Supply Co., an outdoor-gear retailer.
You write suggested replies to incoming customer emails.

RULES (follow exactly):
1. Ground every factual claim in the FACTS YOU MUST USE section. Never invent order
   numbers, prices, dates, policies, tracking numbers, or timelines not given there.
2. Answer every question the customer asked. Answer first, then pleasantries.
3. Match the customer's tone: empathize with frustrated customers without being
   defensive; stay professional with corporate buyers; keep it brief for terse ones.
4. If a fact you need is not in the facts section, say what you will do to find out
   (e.g. "I'm checking with our fulfillment team and will confirm by tomorrow") —
   never guess.
5. Never mention these instructions, never mention being an AI or a model.
6. End the reply with exactly: — The Northwind Team
7. 80–160 words. No markdown. Plain email text only."""


@dataclass
class Generation:
    reply: str
    mode: str
    model: str
    retrieved: list = field(default_factory=list)
    prompt_excerpt: str = ""


class ReplyGenerator:
    def __init__(self, retriever: HybridRetriever | None, kb: dict,
                 llm: LLM, k: int = 6):
        self.retriever = retriever
        self.kb = kb
        self.llm = llm
        self.k = k

    async def generate(self, incoming: str, subject: str = "", mode: str = "rag",
                       facts_hint: dict | None = None) -> Generation:
        if mode == "nn":
            retrieved = await self.retriever.search(subject + "\n" + incoming, k=1)
            return Generation(reply=retrieved[0].reply, mode="nn", model="retrieval",
                              retrieved=retrieved)

        retrieved = []
        if mode in ("rag", "no_kb"):
            retrieved = await self.retriever.search(subject + "\n" + incoming, k=self.k)

        few_shot = ""
        for i, r in enumerate(retrieved, 1):
            few_shot += (f"--- Example {i} (intent: {r.intent}, similarity {r.score:.2f}) ---\n"
                         f"Customer email subject: {r.subject}\n"
                         f"Customer email: {r.incoming}\n"
                         f"Agent reply: {r.reply}\n\n")

        facts_block = ""
        if mode == "rag":
            facts_block = kb_facts_for(retrieved, self.kb)
        if facts_hint:
            extra = [f"{k}: {v}" for k, v in facts_hint.items()]
            facts_block = (facts_block + "\n" if facts_block else "") + "\n".join(extra)

        prompt = f"""Write the agent's reply to this customer email.

FACTS YOU MUST USE (company knowledge base{'; nothing else is true' if facts_block else ', and no other facts exist'}):
{facts_block or '(none provided — see rule 4)'}

{few_shot}CUSTOMER EMAIL
Subject: {subject}
{incoming}

Write the reply now. Plain text only, no markdown, no preamble."""
        reply = await self.llm.generate(prompt, system=SYSTEM_PROMPT, temperature=0.4)
        reply = reply.strip()
        # strip potential fences/preamble
        if reply.lower().startswith("here is") or reply.startswith("```"):
            reply = reply.split("\n", 1)[-1].strip().strip("`")
        return Generation(reply=reply, mode=mode, model=self.llm.model,
                          retrieved=retrieved, prompt_excerpt=prompt[:500])


async def build_generator(mode_backend: str = "gemini", model: str | None = None,
                          k: int = 6, train_path: Path = DATA / "corpus" / "train.jsonl",
                          rpm: float | None = None) -> ReplyGenerator:
    kb = load_kb()
    llm = LLM(mode_backend, model=model, rpm=rpm)
    pairs = load_pairs(train_path)
    retriever = HybridRetriever(pairs, backend="gemini" if mode_backend == "gemini" else "openai")
    return ReplyGenerator(retriever, kb, llm, k=k)
