"""Northwind Supply Co. — synthetic (email, reply) dataset builder.

Design principles
-----------------
1. FACT-ANCHORED: every scenario is generated from a template whose ground-truth
   facts (order number, price, refund window, shipping windows, policy rules)
   are stored in the pair itself as `facts`. The evaluator checks generated
   replies against these facts programmatically - no LLM needed for entity
   and date checks.
2. REPRODUCIBLE: all randomness flows from a single seed recorded in the
   manifest. Re-running the builder with the same seed yields byte-identical
   data.
3. INTENT-BALANCED: 10 scenario families spanning the support mailbox, each
   with sub-type variety, difficulty tiering, and register mix.
4. TEXT REALIZED BY LLM: email/reply text is written by an LLM from the
   structured scenario so the text is natural, then validated by programmatic
   gates (fact survival, forbidden content, length). Failures are regenerated.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import yaml

DATA_DIR = Path(__file__).resolve().parents[3] / "data"

CUSTOMER_NAMES = [
    "Maya Chen", "Liam O'Connor", "Priya Raman", "Diego Sáenz", "Freya Lund",
    "Marcus Bell", "Aiko Tanaka", "Sam Whitfield", "Noor Al-Asadi", "Ethan Kowalski",
    "Rosa Delgado", "Henrik Dahl", "Amara Okafor", "Jonas Petersen", "Leila Haddad",
    "Tomás Ferreira", "Grace Iwu", "Ravi Nair", "Sofia Marchetti", "Ben Osei",
    "Ingrid Solberg", "Kwame Mensah", "Yara Haddad", "Felix Braun", "Nadia Petrov",
    "Owen Gallagher", "Mei-Ling Wu", "Carlos Ruiz", "Hannah Bergström", "Tariq Aziz",
    "Julia Sørensen", "Dev Patel", "Chiara Rossi", "Andre Boucher", "Zoe Lambert",
    "Mateo Silva", "Fatima Zahra", "Rowan Kelly", "Katarzyna Nowak", "Seamus Byrne",
    "Ayesha Siddiqui", "Victor Nunes", "Elin Jonsdottir", "Pablo Herrera", "Mina Toure",
]

REGISTERS = {
    "calm":        "polite, even-tempered, patient customer",
    "frustrated":  "clearly annoyed but civil; mentions this is the second time they have had to ask",
    "furious":     "very angry, threatening a 1-star review and a chargeback; no profanity",
    "terse":       "extremely short, businesslike, zero pleasantries",
    "chatty":      "warm, friendly, one small personal anecdote, an exclamation or two",
    "professional":"formal corporate buyer tone, references their procurement process",
    "grateful":    "effusively thankful for prior help, then asks the new question",
}

INTENTS = [
    "refund_request", "shipping_delay", "defect_warranty", "exchange_size",
    "order_cancel", "order_modify", "product_question", "warranty_denied",
    "price_adjustment", "shipping_options", "gift_card", "complaint_escalation",
]


def load_kb(path: Path | None = None) -> dict:
    with open(path or DATA_DIR / "knowledge_base.yaml") as f:
        return yaml.safe_load(f)


@dataclass
class Scenario:
    intent: str
    sub: str
    register: str
    template_id: str
    customer: str
    email_spec: str          # instructions for the LLM to write the incoming email
    reply_spec: str          # instructions for the LLM to write the agent reply
    facts: dict = field(default_factory=dict)
    kb_anchors: list = field(default_factory=list)
    questions: list = field(default_factory=list)
    actions: list = field(default_factory=list)


class ScenarioBuilder:
    """Programmatic scenario construction: facts first, text spec second."""

    def __init__(self, kb: dict, rng: random.Random):
        self.kb = kb
        self.rng = rng
        self.products = kb["catalog"]

    # ---------- helpers ----------
    def _reg(self) -> str:
        return self.rng.choice(list(REGISTERS.keys()))

    def _name(self) -> str:
        return self.rng.choice(CUSTOMER_NAMES)

    def _order_no(self) -> str:
        return f"{self.rng.randint(10000, 99999)}-{self.rng.randint(1000000, 9999999)}"

    def _pick(self, need_sizes=False, need_colors=False) -> dict:
        pool = self.products
        if need_sizes:
            pool = [p for p in pool if p.get("sizes")]
        if need_colors:
            pool = [p for p in pool if p.get("colors")]
        return self.rng.choice(pool)

    def _days(self, lo=1, hi=30) -> int:
        return self.rng.randint(lo, hi)

    # ---------- intent families ----------

    def refund_request(self) -> Scenario:
        p = self._pick()
        qty = self.rng.choice([1, 1, 1, 2])
        price = round(p["price_usd"] * qty, 2)
        reason = self.rng.choice([
            "wrong size", "did not fit as expected", "found a better price elsewhere",
            "gift the recipient did not like", "ordered two but only needed one",
        ])
        days_since = self._days(1, 12)
        order = self._order_no()
        reg = self._reg()
        name = self._name()
        pol = self.kb["policies"]["returns"]
        facts = {
            "order_number": order, "item": p["name"], "sku": p["sku"],
            "quantity": qty, "price_usd": price,
            "return_window_days": pol["window_days"],
            "refund_window_text": pol["refund_window_text"],
            "restocking_fee_pct": pol["restocking_fee_pct"],
            "customer_pays_return_shipping": "unless defective" in pol["return_shipping"],
            "days_since_delivery": days_since,
            "eligible": days_since <= pol["window_days"],
        }
        email_spec = (
            f"Write an email from {name} to customer support (register: {REGISTERS[reg]}).\n"
            f"Subject: refund request for order {order}\n"
            f"Content requirements: they bought the {p['name']} (qty {qty}, ${price:.2f}) on order {order}; "
            f"delivered {days_since} days ago; reason for return: {reason}. "
            f"They ask for a refund and want to know how long the refund takes. "
            f"MUST include verbatim the order number {order}. Keep it 60-120 words. Do NOT invent other order numbers, prices, or policies."
        )
        reply_spec = (
            f"Write the support agent's reply (warm, concise, honest — brand voice: answers first, pleasantries short).\n"
            f"Required facts: order {order}, {p['name']}, refund amount ${price:.2f} to the original payment method, "
            f"refund appears in {pol['refund_window_text']} after being issued, no restocking fee, "
            f"customer pays return shipping, return must be initiated within {pol['window_days']} days of delivery "
            f"({days_since} days have passed, so it IS still eligible). "
            f"Tone must match: {REGISTERS[reg]}. 80-150 words. Sign off '— The Northwind Team'. "
            f"Do NOT mention policies not listed here. Do NOT invent dates or tracking numbers."
        )
        return Scenario("refund_request", reason, reg, "refund-request-v1", name,
                        email_spec, reply_spec, facts,
                        ["policies.returns", f"catalog.{p['sku']}"],
                        ["requests a refund", "asks how long the refund takes"],
                        ["approve return", "state refund window", "state return-shipping rule"])

    def shipping_delay(self) -> Scenario:
        p = self._pick()
        order = self._order_no()
        reason = self.rng.choice(self.kb["scenario_reasons"]["shipping_delay"])
        days_late = self._days(2, 9)
        reg = self._reg()
        name = self._name()
        ship = self.kb["policies"]["shipping"]
        facts = {
            "order_number": order, "item": p["name"], "sku": p["sku"],
            "delay_cause": reason, "days_late": days_late,
            "standard_days": ship["domestic_standard_days"],
            "tracking_note": ship["tracking_note"],
            "express_fee_usd": ship["express_upgrade_fee_usd"],
        }
        email_spec = (
            f"Write an email from {name} (register: {REGISTERS[reg]}).\n"
            f"Subject: order {order} still not here\n"
            f"Content: their {p['name']} on order {order} was supposed to arrive within "
            f"{ship['domestic_standard_days'][0]}–{ship['domestic_standard_days'][1]} business days but is "
            f"{days_late} days past that; tracking hasn't updated. They ask where it is and when it will arrive. "
            f"MUST include verbatim the order number {order}. 60-120 words. Do NOT invent tracking numbers or ETAs."
        )
        reply_spec = (
            f"Write the support agent's reply (warm, honest, concise).\n"
            f"Required facts: apologize; current status is a {reason} in transit; new ETA is at most "
            f"{days_late} more days; tracking links activate within 24 hours of a carrier scan "
            f"('{ship['tracking_note']}'). Optionally mention express upgrade (${ship['express_upgrade_fee_usd']}) "
            f"for future orders. Order number {order} must appear. Tone matches: {REGISTERS[reg]}. 80-150 words. "
            f"Sign '— The Northwind Team'. Do NOT invent a specific delivery date or tracking number."
        )
        return Scenario("shipping_delay", reason, reg, "shipping-delay-v1", name,
                        email_spec, reply_spec, facts,
                        ["policies.shipping"],
                        ["asks where the package is", "asks when it will arrive"],
                        ["apologize", "state cause", "give ETA window"])

    def defect_warranty(self) -> Scenario:
        p = self._pick()
        order = self._order_no()
        defect = self.rng.choice(self.kb["scenario_reasons"]["defect_types"])
        months_owned = self._days(1, 20)
        reg = self._reg()
        name = self._name()
        war = self.kb["policies"]["warranty"]
        facts = {
            "order_number": order, "item": p["name"], "sku": p["sku"],
            "defect_type": defect, "months_owned": months_owned,
            "warranty_months": war["duration_months"],
            "in_warranty": months_owned <= war["duration_months"],
            "replacement_before_return": "replacement before return" in war["process"],
            "photo_required": "photo" in war["process"],
        }
        email_spec = (
            f"Write an email from {name} (register: {REGISTERS[reg]}).\n"
            f"Subject: {defect} on my {p['name']}\n"
            f"Content: owned the {p['name']} (order {order}) for about {months_owned} months; "
            f"the {defect} appeared last week. They ask how to get a replacement and whether "
            f"they must send the item back first. MUST include verbatim {order}. 60-120 words. "
            f"Do NOT mention warranties or timelines beyond their own experience."
        )
        reply_spec = (
            f"Write the support agent's reply (warm, concise).\n"
            f"Required facts: the {war['duration_months']}-month warranty covers {defect} and "
            f"{months_owned} months is within it; ask for a photo of the defect plus the order number "
            f"({order}); confirm we ship the replacement BEFORE they return the defective unit. "
            f"Tone matches: {REGISTERS[reg]}. 80-150 words. Sign '— The Northwind Team'. "
            f"Do NOT promise a refund; the resolution path is a replacement."
        )
        return Scenario("defect_warranty", defect, reg, "defect-warranty-v1", name,
                        email_spec, reply_spec, facts,
                        ["policies.warranty", f"catalog.{p['sku']}"],
                        ["asks how to get a replacement", "asks whether they must return the item first"],
                        ["confirm warranty coverage", "request photo + order number", "commit to replacement-first"])

    def exchange_size(self) -> Scenario:
        p = self._pick(need_sizes=True)
        order = self._order_no()
        sizes = p["sizes"]
        from_size = self.rng.choice(sizes)
        to_size = self.rng.choice([s for s in sizes if s != from_size])
        reg = self._reg()
        name = self._name()
        ret = self.kb["policies"]["returns"]
        facts = {
            "order_number": order, "item": p["name"], "sku": p["sku"],
            "from_size": from_size, "to_size": to_size,
            "exchange_shipping_covered": True,
            "return_window_days": ret["window_days"],
        }
        email_spec = (
            f"Write an email from {name} (register: {REGISTERS[reg]}).\n"
            f"Subject: size exchange — order {order}\n"
            f"Content: the {p['name']} on order {order} in size {from_size} "
            f"{'runs too small' if to_size > from_size else 'runs too large'}. They ask to exchange for "
            f"size {to_size} and who pays shipping. MUST include verbatim {order}. 60-120 words."
        )
        reply_spec = (
            f"Write the support agent's reply (warm, concise).\n"
            f"Required facts: exchange approved from {from_size} to {to_size} on order {order}; "
            f"Northwind covers shipping on size/color exchanges (customer does not pay); "
            f"the 30-day window applies but is not in danger. Tone matches: {REGISTERS[reg]}. "
            f"80-150 words. Sign '— The Northwind Team'. Do NOT ask them to pay for shipping."
        )
        return Scenario("exchange_size", f"{from_size}->{to_size}", reg, "exchange-size-v1", name,
                        email_spec, reply_spec, facts,
                        ["policies.returns.exchange_shipping", f"catalog.{p['sku']}"],
                        ["requests size exchange", "asks who pays shipping"],
                        ["approve exchange", "state Northwind pays exchange shipping"])

    def order_cancel(self) -> Scenario:
        p = self._pick()
        order = self._order_no()
        shipped = self.rng.random() < 0.3
        reg = self._reg()
        name = self._name()
        ord_pol = self.kb["policies"]["orders"]
        facts = {
            "order_number": order, "item": p["name"], "sku": p["sku"],
            "price_usd": p["price_usd"],
            "already_shipped": shipped,
            "cancellation_allowed": (not shipped) and ord_pol["cancellation_before_ship"],
            "refund_window_text": self.kb["policies"]["returns"]["refund_window_text"],
        }
        email_spec = (
            f"Write an email from {name} (register: {REGISTERS[reg]}).\n"
            f"Subject: cancel my order {order}\n"
            f"Content: they want to cancel the {p['name']} (${p['price_usd']:.2f}) on order {order}; "
            f"they changed their mind / found it cheaper locally. "
            + ("NOTE: unknown to the customer, the package has already shipped." if shipped else "The order is still being prepared, unknown to the customer.")
            + f" MUST include verbatim {order}. 60-110 words. Do NOT state whether it shipped."
        )
        reply_spec = (
            f"Write the support agent's reply (warm, honest, concise).\n"
            + (f"Required facts: the order HAS shipped, so it cannot be cancelled; "
               f"instead offer the return path: refuse delivery or return it within the 30-day window for a "
               f"full refund to the original payment method ({self.kb['policies']['returns']['refund_window_text']})."
               if shipped else
               "Required facts: the order has NOT shipped, so the cancellation is done; refund of "
               f"${p['price_usd']:.2f} to the original payment method in "
               f"{self.kb['policies']['returns']['refund_window_text']}.")
            + f" Order number {order} must appear. Tone matches: {REGISTERS[reg]}. 80-140 words. "
            f"Sign '— The Northwind Team'. Do NOT invent timelines beyond the refund window."
        )
        return Scenario("order_cancel", "shipped" if shipped else "pre-ship", reg, "order-cancel-v1", name,
                        email_spec, reply_spec, facts,
                        ["policies.orders", "policies.returns"],
                        ["requests cancellation", "implicitly asks what happens to their money"],
                        ["state whether cancellation is possible", "state the refund path"])

    def order_modify(self) -> Scenario:
        p = self._pick()
        order = self._order_no()
        hours_since = self.rng.choice([1, 2, 3, 6, 24])
        within_window = hours_since <= self.kb["policies"]["orders"]["modification_window_hours"]
        reg = self._reg()
        name = self._name()
        facts = {
            "order_number": order, "item": p["name"], "sku": p["sku"],
            "hours_since_order": hours_since,
            "modification_window_hours": self.kb["policies"]["orders"]["modification_window_hours"],
            "modifiable": within_window,
            "address_change_policy": self.kb["policies"]["orders"]["address_change"],
        }
        email_spec = (
            f"Write an email from {name} (register: {REGISTERS[reg]}).\n"
            f"Subject: change address on order {order}\n"
            f"Content: they placed order {order} ({p['name']}) {hours_since} hours ago and just realized "
            f"the shipping address is wrong (old apartment / typo). They ask to fix the address. "
            f"MUST include verbatim {order}. 50-110 words."
        )
        reply_spec = (
            f"Write the support agent's reply (warm, concise).\n"
            + ("Required facts: order was placed within the "
               f"{self.kb['policies']['orders']['modification_window_hours']}-hour modification window "
               "({hours_since} hours ago), so the address CAN be changed; ask them to confirm the new address."
               .format(hours_since=hours_since)
               if within_window else
               "Required facts: the order was placed "
               f"{hours_since} hours ago, past the "
               f"{self.kb['policies']['orders']['modification_window_hours']}-hour modification window, but "
               f"address changes are possible any time before the package is scanned by the carrier; ask them "
               f"to confirm the new address so we can update it.")
            + f" Order number {order} must appear. Tone matches: {REGISTERS[reg]}. 70-130 words. "
            f"Sign '— The Northwind Team'."
        )
        return Scenario("order_modify", "within-window" if within_window else "past-window", reg,
                        "order-modify-v1", name, email_spec, reply_spec, facts,
                        ["policies.orders"],
                        ["requests an address change"],
                        ["state whether the change is possible", "ask for the corrected address"])

    def product_question(self) -> Scenario:
        p = self._pick()
        reg = self._reg()
        name = self._name()
        asks = self.rng.choice(["weight", "sizes", "colors", "stock", "shipping_time"])
        if asks == "weight":
            attr, ans = "weight", f"{p.get('weight_kg', 'the listed weight')} kg"
            q = "how much it weighs"
        elif asks == "sizes" and p.get("sizes"):
            attr, ans = "sizes", ", ".join(map(str, p["sizes"]))
            q = "what sizes are available"
        elif asks == "colors" and p.get("colors"):
            attr, ans = "colors", ", ".join(p["colors"])
            q = "what colors it comes in"
        elif asks == "stock":
            attr, ans = "stock", p["stock_status"].replace("_", " ")
            q = "whether it is in stock"
        else:
            attr, ans = "shipping", f"{self.kb['policies']['shipping']['domestic_standard_days'][0]}–{self.kb['policies']['shipping']['domestic_standard_days'][1]} business days standard"
            q = "how long shipping takes"
        facts = {
            "item": p["name"], "sku": p["sku"], "asked_attribute": attr,
            "price_usd": p["price_usd"], "answer": ans,
            "free_shipping_threshold_usd": self.kb["policies"]["shipping"]["free_shipping_threshold_usd"],
        }
        email_spec = (
            f"Write an email from {name} (register: {REGISTERS[reg]}).\n"
            f"Subject: question about the {p['name']}\n"
            f"Content: they are considering buying the {p['name']} (${p['price_usd']:.2f}) and ask {q}. "
            f"60-110 words. Do NOT include the answer; they don't know it. Do NOT invent order numbers."
        )
        reply_spec = (
            f"Write the support agent's reply (warm, concise, answers first).\n"
            f"Required facts: answer the question directly — {q.replace('how much', 'it weighs').replace('what sizes', 'sizes available are').replace('what colors', 'colors available are').replace('whether it is in stock', 'stock status is').replace('how long shipping takes', 'standard shipping is')} "
            f"'{ans}'. Price ${p['price_usd']:.2f}. "
            f"Also mention free standard shipping over ${self.kb['policies']['shipping']['free_shipping_threshold_usd']}. "
            f"Tone matches: {REGISTERS[reg]}. 70-130 words. Sign '— The Northwind Team'. "
            f"Do NOT invent specs beyond: {ans}."
        )
        return Scenario("product_question", attr, reg, "product-question-v1", name,
                        email_spec, reply_spec, facts,
                        [f"catalog.{p['sku']}", "policies.shipping"],
                        [f"asks {q}"],
                        ["answer the product question", "mention free-shipping threshold"])

    def warranty_denied(self) -> Scenario:
        """Hard case: customer assumes warranty applies, but it doesn't."""
        p = self._pick()
        order = self._order_no()
        months_owned = self.rng.randint(25, 40)
        not_covered = self.rng.choice(self.kb["policies"]["warranty"]["not_covered"])
        reg = self.rng.choice(["frustrated", "furious", "professional"])
        name = self._name()
        war = self.kb["policies"]["warranty"]
        facts = {
            "order_number": order, "item": p["name"], "sku": p["sku"],
            "months_owned": months_owned, "warranty_months": war["duration_months"],
            "in_warranty": False, "not_covered_reason": not_covered,
        }
        email_spec = (
            f"Write an email from {name} (register: {REGISTERS[reg]}).\n"
            f"Subject: replacement for my {p['name']} — order {order}\n"
            f"Content: they've had the {p['name']} for {months_owned} months and it broke "
            f"({not_covered}); they believe the warranty should cover it and want a replacement shipped now. "
            f"They mention a friend got a free replacement for a similar issue. "
            f"MUST include verbatim {order}. 70-130 words."
        )
        reply_spec = (
            f"Write the support agent's reply (warm but HONEST — must decline gracefully).\n"
            f"Required facts: politely decline the warranty claim because the warranty is "
            f"{war['duration_months']} months and the item is {months_owned} months old, and because "
            f"'{not_covered}' is not covered. Do NOT offer a free replacement. Empathize genuinely. "
            f"Offer an honest alternative: they can return it under the 30-day window if it's within "
            f"{self.kb['policies']['returns']['window_days']} days of delivery ONLY IF true — it is NOT (it's "
            f"{months_owned} months old), so instead offer a discount on a replacement: mention that replacement "
            f"parts or repair guidance can be shared. Order {order} must appear. "
            f"Tone: defuse, don't grovel. 90-160 words. Sign '— The Northwind Team'. "
            f"Do NOT invent a discount percentage."
        )
        return Scenario("warranty_denied", not_covered, reg, "warranty-denied-v1", name,
                        email_spec, reply_spec, facts,
                        ["policies.warranty"],
                        ["demands a warranty replacement"],
                        ["decline honestly with reason", "empathize", "offer honest alternative"])

    def price_adjustment(self) -> Scenario:
        p = self._pick()
        order = self._order_no()
        days_since = self.rng.randint(1, 30)
        within = days_since <= self.kb["policies"]["orders"]["price_adjustment_days"]
        new_price = round(p["price_usd"] * self.rng.uniform(0.75, 0.95), 2)
        reg = self._reg()
        name = self._name()
        facts = {
            "order_number": order, "item": p["name"], "sku": p["sku"],
            "paid_usd": p["price_usd"], "new_price_usd": new_price,
            "days_since_purchase": days_since,
            "adjustment_window_days": self.kb["policies"]["orders"]["price_adjustment_days"],
            "adjustment_approved": within,
        }
        email_spec = (
            f"Write an email from {name} (register: {REGISTERS[reg]}).\n"
            f"Subject: price dropped after my purchase — order {order}\n"
            f"Content: they bought the {p['name']} at ${p['price_usd']:.2f} ({days_since} days ago, order {order}) "
            f"and now see it listed at ${new_price:.2f}; they ask for the difference back. "
            f"MUST include verbatim {order} and both prices. 60-120 words."
        )
        reply_spec = (
            f"Write the support agent's reply (warm, honest).\n"
            + (f"Required facts: the price-adjustment window is "
               f"{self.kb['policies']['orders']['price_adjustment_days']} days and the purchase was "
               f"{days_since} days ago, so it qualifies: refund the difference of "
               f"${round(p['price_usd'] - new_price, 2):.2f} to the original payment method in "
               f"{self.kb['policies']['returns']['refund_window_text']}."
               if within else
               f"Required facts: the price-adjustment window is "
               f"{self.kb['policies']['orders']['price_adjustment_days']} days; the purchase was "
               f"{days_since} days ago, so it no longer qualifies. Politely explain, empathize, "
               f"and mention that current-sale prices apply to new orders only.")
            + f" Order {order} must appear. Tone matches: {REGISTERS[reg]}. 80-150 words. "
            f"Sign '— The Northwind Team'."
        )
        return Scenario("price_adjustment", "within" if within else "outside-window", reg,
                        "price-adjustment-v1", name, email_spec, reply_spec, facts,
                        ["policies.orders"],
                        ["asks for a price difference refund"],
                        ["state whether the adjustment applies", "state the amount or reason"])

    def shipping_options(self) -> Scenario:
        p = self._pick()
        order = self._order_no()
        reg = self._reg()
        name = self._name()
        ship = self.kb["policies"]["shipping"]
        want_express = self.rng.random() < 0.5
        facts = {
            "order_number": order, "item": p["name"], "sku": p["sku"],
            "standard_days": ship["domestic_standard_days"],
            "express_days": ship["domestic_express_days"],
            "express_fee_usd": ship["express_upgrade_fee_usd"],
            "free_threshold_usd": ship["free_shipping_threshold_usd"],
            "cutoff_time": ship["cutoff_time"],
        }
        email_spec = (
            f"Write an email from {name} (register: {REGISTERS[reg]}).\n"
            f"Subject: can I upgrade shipping on order {order}?\n"
            f"Content: they need the {p['name']} (order {order}) "
            + ("by this weekend for a trip" if want_express else "and ask what shipping options exist and what they cost")
            + "; they ask whether shipping can be upgraded and what it costs. "
            f"MUST include verbatim {order}. 60-110 words."
        )
        reply_spec = (
            f"Write the support agent's reply (warm, concise).\n"
            f"Required facts: express shipping is {ship['domestic_express_days'][0]}–{ship['domestic_express_days'][1]} "
            f"business days for a ${ship['express_upgrade_fee_usd']} upgrade; standard is "
            f"{ship['domestic_standard_days'][0]}–{ship['domestic_standard_days'][1]} business days "
            f"(free over ${ship['free_shipping_threshold_usd']}); orders placed before "
            f"{ship['cutoff_time']} ship the same day. Offer to apply the upgrade to order {order}. "
            f"Tone matches: {REGISTERS[reg]}. 80-140 words. Sign '— The Northwind Team'. "
            f"Do NOT invent other fees."
        )
        return Scenario("shipping_options", "express" if want_express else "general", reg,
                        "shipping-options-v1", name, email_spec, reply_spec, facts,
                        ["policies.shipping"],
                        ["asks about shipping upgrade", "asks about cost"],
                        ["state express fee and window", "state cutoff time", "offer to apply upgrade"])

    def gift_card(self) -> Scenario:
        p = self._pick()
        order = self._order_no()
        reg = self._reg()
        name = self._name()
        gc = self.kb["policies"]["gift_cards"]
        facts = {
            "order_number": order,
            "expiry_months": gc["expiry_monthly"] if "expiry_monthly" in gc else gc.get("expiry_months", 24),
            "combinable": gc["combinable_with_discounts"],
            "physical_card_shipping": gc["physical_card_shipping"],
        }
        email_spec = (
            f"Write an email from {name} (register: {REGISTERS[reg]}).\n"
            f"Subject: gift card question\n"
            f"Content: they want to buy a gift card for their niece's birthday and ask "
            f"how long gift cards last, whether gift cards can be combined with a discount code, "
            f"and how physical cards ship. 60-110 words. No order number needed (they haven't ordered yet), "
            f"but if they mention a previous order, use {order}."
        )
        reply_spec = (
            f"Write the support agent's reply (warm, concise).\n"
            f"Required facts: gift cards are valid for {facts['expiry_months']} months; "
            f"they CAN be combined with discount codes ({gc['combinable_with_discounts']}); "
            f"physical cards ship free via standard mail ({gc['physical_card_shipping']}). "
            f"Tone matches: {REGISTERS[reg]}. 70-130 words. Sign '— The Northwind Team'. "
            f"Do NOT invent gift card denominations."
        )
        return Scenario("gift_card", "general", reg, "gift-card-v1", name,
                        email_spec, reply_spec, facts,
                        ["policies.gift_cards"],
                        ["asks gift card expiry", "asks combining with discounts", "asks about physical card shipping"],
                        ["state expiry", "state combinability", "state physical card shipping"])

    def complaint_escalation(self) -> Scenario:
        """Angry repeat customer — the metric should reward de-escalation."""
        p = self._pick()
        order = self._order_no()
        reg = self.rng.choice(["furious", "frustrated"])
        name = self._name()
        prior_contacts = self.rng.randint(2, 4)
        issue = self.rng.choice(["wrong item shipped", "package arrived damaged", "missing item from order"])
        goodwill = self.rng.choice([10, 15, 20])
        facts = {
            "order_number": order, "item": p["name"], "sku": p["sku"],
            "issue": issue, "prior_contacts": prior_contacts,
            "goodwill_discount_pct": goodwill,
            "resolution": "replacement shipped express, no charge",
        }
        email_spec = (
            f"Write an email from {name} (register: {REGISTERS[reg]}).\n"
            f"Subject: this is unacceptable — order {order}\n"
            f"Content: this is their {ordinal(prior_contacts)} email about order {order}; "
            f"the issue is: {issue}. They are done with apologies and want it fixed. "
            f"They threaten to dispute the charge with their bank. "
            f"MUST include verbatim {order}. 80-150 words."
        )
        reply_spec = (
            f"Write the support agent's reply (calm, accountable, zero defensiveness — DE-ESCALATE).\n"
            f"Required facts: acknowledge the {prior_contacts} prior contacts without excuses; "
            f"commit to a concrete fix: {facts['resolution']} for the {issue} on order {order}; "
            f"add a {goodwill}% discount code for their next order; give a direct follow-up commitment "
            f"(we will confirm within one business day). Tone: own the failure, no groveling, no blaming the carrier. "
            f"90-170 words. Sign '— The Northwind Team'. Do NOT promise refunds of the full order."
        )
        return Scenario("complaint_escalation", issue, reg, "complaint-escalation-v1", name,
                        email_spec, reply_spec, facts,
                        ["policies.shipping", "policies.returns"],
                        ["demands the problem actually be fixed", "expects acknowledgment of repeat contact"],
                        ["acknowledge repeat contact", "commit to concrete fix", "offer goodwill discount"])


def ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suf = "th"
    else:
        suf = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


BUILDERS = {
    "refund_request": ScenarioBuilder.refund_request,
    "shipping_delay": ScenarioBuilder.shipping_delay,
    "defect_warranty": ScenarioBuilder.defect_warranty,
    "exchange_size": ScenarioBuilder.exchange_size,
    "order_cancel": ScenarioBuilder.order_cancel,
    "order_modify": ScenarioBuilder.order_modify,
    "product_question": ScenarioBuilder.product_question,
    "warranty_denied": ScenarioBuilder.warranty_denied,
    "price_adjustment": ScenarioBuilder.price_adjustment,
    "shipping_options": ScenarioBuilder.shipping_options,
    "gift_card": ScenarioBuilder.gift_card,
    "complaint_escalation": ScenarioBuilder.complaint_escalation,
}


def build_scenarios(seed: int, n_per_intent: int = 34, kb: dict | None = None) -> list[Scenario]:
    """Deterministically build the scenario skeleton for all 12 intent families."""
    rng = random.Random(seed)
    kb = kb or load_kb()
    sb = ScenarioBuilder(kb, rng)
    out = []
    for intent, builder in BUILDERS.items():
        for _ in range(n_per_intent):
            out.append(builder(sb))
    return out
