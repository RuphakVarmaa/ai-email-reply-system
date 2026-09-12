"""Programmatic fact checks — the deterministic backbone of the accuracy system.

No LLM involved. These extract checkable entities from a reply and compare them
to the scenario's ground-truth facts dict + the KB. Each check returns a verdict:
  PASS / FAIL / NOT_APPLICABLE, with evidence strings.

Design principle: an LLM judge can be fooled; a regex on the order number cannot.
Deterministic checks carry the heaviest evidentiary weight, LLM judgments fill
the semantic gaps (coverage, tone, action correctness).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

ORDER_RE = re.compile(r"\b(\d{5})-(\d{7})\b")
MONEY_RE = re.compile(r"\$\s?([\d,]+(?:\.\d{1,2})?)")
DAY_RANGE_RE = re.compile(r"(\d+)\s*[–—-]\s*(\d+)\s*(?:business\s*)?days?", re.I)
NDAYS_RE = re.compile(r"(\d+)\s*(?:business\s*)?days", re.I)
PCT_RE = re.compile(r"(\d{1,3})\s*%")


@dataclass
class Check:
    name: str
    verdict: str  # PASS / FAIL / NOT_APPLICABLE
    detail: str
    severity: str = "major"  # major | minor


@dataclass
class FactReport:
    checks: list = field(default_factory=list)

    @property
    def n_pass(self):
        return sum(1 for c in self.checks if c.verdict == "PASS")

    @property
    def n_fail(self):
        return sum(1 for c in self.checks if c.verdict == "FAIL")

    @property
    def n_applicable(self):
        return sum(1 for c in self.checks if c.verdict != "NOT_APPLICABLE")

    @property
    def entity_accuracy(self):
        if self.n_applicable == 0:
            return None
        return self.n_pass / self.n_applicable


def _normalize_money(s: str) -> float:
    return float(s.replace(",", ""))


def check_entity_survival(reply: str, facts: dict) -> list[Check]:
    """The reply must carry over the correct order number, price, size, item name."""
    checks = []
    order = facts.get("order_number")
    if order:
        m = ORDER_RE.search(reply)
        if m and f"{m.group(1)}-{m.group(2)}" == order:
            checks.append(Check("order_number", "PASS", f"reply cites order {order}"))
        elif m:
            checks.append(Check("order_number", "FAIL",
                                f"reply cites WRONG order {m.group(1)}-{m.group(2)} (expected {order})", "major"))
        else:
            checks.append(Check("order_number", "FAIL",
                                f"reply omits the order number ({order})", "minor"))
    price = facts.get("price_usd") or facts.get("paid_usd")
    if price:
        if any(abs(_normalize_money(m) - float(price)) < 0.005 for m in MONEY_RE.findall(reply)):
            checks.append(Check("price", "PASS", f"reply cites correct amount ${price:.2f}"))
        else:
            checks.append(Check("price", "FAIL",
                                f"reply does not state the correct amount ${price:.2f}", "minor"))
    # size exchange correctness
    if "to_size" in facts and "from_size" in facts:
        to, frm = str(facts["to_size"]), str(facts["from_size"])
        has_to = re.search(rf"\b{re.escape(to)}\b", reply)
        has_from = re.search(rf"\b{re.escape(frm)}\b", reply)
        if has_to and (has_from or True):
            checks.append(Check("exchange_sizes", "PASS",
                                f"reply references exchange {frm} -> {to}"))
        else:
            checks.append(Check("exchange_sizes", "FAIL",
                                f"reply does not clearly state the exchange {frm} -> {to}", "minor"))
    item = facts.get("item")
    if item:
        first_words = " ".join(item.split()[:3])
        if first_words.lower() in reply.lower():
            checks.append(Check("item_reference", "PASS", f"reply references the item '{item}'"))
        else:
            checks.append(Check("item_reference", "FAIL",
                                f"reply never mentions the item '{item}'", "minor"))
    return checks


def check_policy_numbers(reply: str, facts: dict, kb: dict) -> list[Check]:
    """Numeric policy claims (windows, fees, thresholds) must match KB."""
    checks = []
    ret = kb["policies"]["returns"]
    ship = kb["policies"]["shipping"]
    war = kb["policies"]["warranty"]

    # Any "X–Y business days" refund-window claim must equal KB refund window text
    refund_text = ret["refund_window_text"]
    m = re.search(r"(\d+)\s*[–—-]\s*(\d+)\s*(?:business\s*)?days", refund_text)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        for dm in DAY_RANGE_RE.finditer(reply):
            if "refund" in reply[max(0, dm.start()-60):dm.start()] or True:
                # a day-range near "refund" mentions
                seg = reply[max(0, dm.start()-80):dm.end()+20]
                if re.search(r"refund|money back|reimburse", seg, re.I):
                    if (int(dm.group(1)), int(dm.group(2))) != (lo, hi):
                        checks.append(Check("refund_window", "FAIL",
                                            f"reply claims refund window {dm.group(0)} but KB says {refund_text}", "major"))
                    else:
                        checks.append(Check("refund_window", "PASS",
                                            f"reply states correct refund window {refund_text}"))
                    break
    # single-day refund claims near 'refund'
    for nd in NDAYS_RE.finditer(reply):
        seg = reply[max(0, nd.start()-80):nd.end()+30]
        if re.search(r"refund|money back|reimburse", seg, re.I):
            n = int(nd.group(1))
            if n > hi + 2:
                checks.append(Check("refund_window", "FAIL",
                                    f"reply claims refund takes {n} days; KB says {refund_text}", "major"))
            break
    # express fee
    fee = ship["express_upgrade_fee_usd"]
    if re.search(r"express", reply, re.I):
        mfee = [float(x) for x in re.findall(r"\$\s?([\d,]+(?:\.\d{1,2})?)", reply)]
        fees_mentioned = [x for x in mfee if abs(x - fee) < 0.005]
        if len(mfee) > 0 and not fees_mentioned and any(abs(x - fee) > 0.01 for x in mfee):
            pass  # other fees may legitimately appear (item price etc.) — not a fee claim
        if re.search(rf"upgrade[^\$]{{0,40}}\$\s?{str(fee).rstrip('0').rstrip('.')}", reply) or \
           re.search(rf"\$\s?{str(fee).rstrip('0').rstrip('.')}[^\w]{{0,20}}(upgrade|express)", reply, re.I):
            checks.append(Check("express_fee", "PASS", f"express fee ${fee} correct"))
    # warranty duration
    if re.search(r"warrant", reply, re.I):
        mo = re.search(r"(\d{1,2})\s*-?\s*month", reply, re.I)
        if mo and int(mo.group(1)) != war["duration_months"]:
            checks.append(Check("warranty_duration", "FAIL",
                                f"reply claims {mo.group(1)}-month warranty; KB says {war['duration_months']}", "major"))
        elif mo:
            checks.append(Check("warranty_duration", "PASS",
                                f"reply states correct {war['duration_months']}-month warranty"))
    # return window
    if re.search(r"return|exchange", reply, re.I):
        mo = re.search(r"(\d{1,2})\s*-?\s*day\s*(?:window|from delivery|return)", reply, re.I)
        if mo and int(mo.group(1)) != ret["window_days"]:
            checks.append(Check("return_window", "FAIL",
                                f"reply claims {mo.group(1)}-day return window; KB says {ret['window_days']}", "major"))
        elif mo:
            checks.append(Check("return_window", "PASS",
                                f"reply states correct {ret['window_days']}-day return window"))
    # free shipping threshold
    if re.search(r"free\s+shipping|shipping\s+free", reply, re.I):
        mo = re.search(r"(?:over|above|more than)\s*\$?\s*([\d,]+)", reply, re.I)
        if mo and int(float(mo.group(1).replace(",", ""))) != ship["free_shipping_threshold_usd"]:
            checks.append(Check("free_shipping_threshold", "FAIL",
                                f"reply claims free shipping over ${mo.group(1)}; KB says ${ship['free_shipping_threshold_usd']}", "major"))
        elif mo:
            checks.append(Check("free_shipping_threshold", "PASS", "threshold correct"))
    return checks


def check_eligibility_consistency(reply: str, facts: dict) -> list[Check]:
    """The reply's decision must match the scenario's ground-truth decision.

    This is where corruption in the perturbation suite is caught: e.g. ground
    truth says warranty is VOID but the reply promises a replacement.
    """
    checks = []
    # warranty void
    if "in_warranty" in facts:
        if facts["in_warranty"] is False:
            if re.search(r"we(?:'ll| will| can)[^.]{0,40}(?:replacement|replace it|ship you a new)", reply, re.I) \
               and not re.search(r"cannot|can't|unable|not able|won't be able", reply, re.I):
                checks.append(Check("warranty_decision", "FAIL",
                                    "ground truth: warranty VOID, but reply promises a replacement", "major"))
            else:
                checks.append(Check("warranty_decision", "PASS",
                                    "reply correctly declines (warranty void)"))
        elif facts["in_warranty"] is True:
            if re.search(r"cannot|can't|unable|not able|outside the warranty|past the warranty", reply, re.I):
                checks.append(Check("warranty_decision", "FAIL",
                                    "ground truth: warranty VALID, but reply declines it", "major"))
            else:
                checks.append(Check("warranty_decision", "PASS",
                                    "reply correctly honors warranty"))
    # cancellation when already shipped
    if "already_shipped" in facts:
        if facts["already_shipped"]:
            if re.search(r"(?:cancel(?:led|ation)? (?:is|has been)?\s*(?:complete|done|processed|confirmed))", reply, re.I) \
               and not re.search(r"cannot|can't|unable|already shipped|too late", reply, re.I):
                checks.append(Check("cancellation_decision", "FAIL",
                                    "order already shipped; reply falsely confirms cancellation", "major"))
            else:
                checks.append(Check("cancellation_decision", "PASS",
                                    "reply handles shipped-order cancellation correctly"))
        else:
            if re.search(r"cannot cancel|can't cancel|unable to cancel|already shipped", reply, re.I):
                checks.append(Check("cancellation_decision", "FAIL",
                                    "order NOT shipped; reply refuses a valid cancellation", "major"))
            else:
                checks.append(Check("cancellation_decision", "PASS",
                                    "reply confirms cancellation (correct)"))
    # price adjustment window
    if "adjustment_approved" in facts:
        amount = None
        if facts.get("paid_usd") and facts.get("new_price_usd"):
            amount = round(facts["paid_usd"] - facts["new_price_usd"], 2)
        if facts["adjustment_approved"] is True:
            if re.search(r"outside the (?:\d+[- ])?day|no longer (?:qualifies|eligible)|cannot (?:offer|refund|honor)", reply, re.I):
                checks.append(Check("price_adjustment_decision", "FAIL",
                                    "ground truth: adjustment QUALIFIES; reply denies it", "major"))
            else:
                ok = True
                detail = "reply approves adjustment (correct)"
                if amount is not None:
                    moneys = [float(x.replace(",", "")) for x in MONEY_RE.findall(reply)]
                    if moneys and not any(abs(x - amount) < 0.01 for x in moneys):
                        # did it state some other amount? suspicious but not disqualifying alone
                        detail = "reply approves adjustment; verify amount mentioned"
                    else:
                        detail = f"reply approves adjustment with correct amount ${amount:.2f}"
                checks.append(Check("price_adjustment_decision", "PASS", detail))
        else:
            if re.search(r"(?:refund|credit|issued)[^.]{0,50}\$|we(?:'ll| will) refund the difference", reply, re.I):
                checks.append(Check("price_adjustment_decision", "FAIL",
                                    "ground truth: adjustment OUTSIDE window; reply grants it anyway", "major"))
            else:
                checks.append(Check("price_adjustment_decision", "PASS",
                                    "reply correctly declines adjustment"))
    # refund eligibility
    if "eligible" in facts:
        if facts["eligible"] is False and re.search(r"full refund|issue(?:d)? (?:a )?refund|we(?:'ll| will) refund", reply, re.I) \
           and not re.search(r"cannot|can't|unable|not eligible|outside", reply, re.I):
            checks.append(Check("refund_eligibility", "FAIL",
                                "ground truth: NOT eligible, but reply promises a refund", "major"))
        elif facts["eligible"] is True:
            checks.append(Check("refund_eligibility", "PASS", "reply proceeds with refund (eligible)"))
    return checks


def check_gates(reply: str, kb: dict) -> list[Check]:
    """Safety gates: forbidden phrases, prompt leak, signature, length."""
    checks = []
    for ph in kb["forbidden_reply_phrases"]:
        if ph.lower() in reply.lower():
            checks.append(Check("forbidden_phrase", "FAIL",
                                f"reply contains forbidden phrase '{ph}'", "major"))
    if re.search(r"as an AI|language model|these instructions|system prompt", reply, re.I):
        checks.append(Check("prompt_leak", "FAIL", "reply leaks AI/instruction framing", "major"))
    if "The Northwind Team" not in reply:
        checks.append(Check("signature", "FAIL", "reply missing '— The Northwind Team' signature", "minor"))
    wc = len(reply.split())
    if wc < 25:
        checks.append(Check("length", "FAIL", f"reply too short ({wc} words)", "minor"))
    elif wc > 260:
        checks.append(Check("length", "FAIL", f"reply too long ({wc} words)", "minor"))
    return checks


def run_fact_checks(reply: str, facts: dict, kb: dict) -> FactReport:
    checks = []
    checks += check_entity_survival(reply, facts)
    checks += check_policy_numbers(reply, facts, kb)
    checks += check_eligibility_consistency(reply, facts)
    checks += check_gates(reply, kb)
    return FactReport(checks=checks)
