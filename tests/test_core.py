"""Unit tests for the email reply system.

Designed to run WITHOUT API calls (mock LLM backend) so they pass in CI.
"""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
import pytest_asyncio

from email_reply.dataset.synth import build_scenarios, load_kb, ScenarioBuilder, REGISTERS
from email_reply.llm import LLM, MockLLM, extract_json
from email_reply.eval.facts import (
    run_fact_checks, check_entity_survival, check_policy_numbers,
    check_eligibility_consistency, check_gates,
)
from email_reply.eval.composite import score_response, overall_scores, tone_norm, WEIGHTS
from email_reply.eval.validate import spearman, pearson, PERTURBATIONS


# ---------------------------------------------------------------- fixtures

@pytest.fixture
def kb():
    return load_kb()


@pytest.fixture
def sample_scenario(kb):
    import random
    sb = ScenarioBuilder(kb, random.Random(42))
    return sb.refund_request()


@pytest.fixture
def good_reply():
    return (
        "Hi Maya, thanks for reaching out about order 39256-3341057. "
        "We're happy to help with your return of the Ridgeline -20°C Sleeping Bag ($199.00). "
        "Since it's been 4 days since delivery, you're well within our 30-day return window. "
        "We'll process your refund of $199.00 to your original payment method, and it should "
        "appear within 5–7 business days after we receive the item back. Please note that "
        "return shipping is on you (no restocking fee, though!). "
        "Let us know if you need a return label. — The Northwind Team"
    )


@pytest.fixture
def bad_reply():
    return (
        "Per our company policy, all requests are handled in order received. "
        "Your refund of $199.00 for order 99999-0000000 will arrive in 30 business days. "
        "As an AI, I cannot guarantee timelines. — The Northwind Team"
    )


# ---------------------------------------------------------------- dataset tests

def test_scenario_builder_produces_all_intents(kb):
    scenarios = build_scenarios(seed=1, n_per_intent=2, kb=kb)
    intents = {s.intent for s in scenarios}
    assert len(intents) >= 10, f"Expected >=10 intents, got {len(intents)}: {intents}"


def test_scenario_facts_have_required_keys(kb):
    import random
    sb = ScenarioBuilder(kb, random.Random(7))
    sc = sb.refund_request()
    assert "order_number" in sc.facts
    assert "price_usd" in sc.facts
    assert "return_window_days" in sc.facts


def test_scenario_reproducibility(kb):
    s1 = build_scenarios(seed=42, n_per_intent=3, kb=kb)
    s2 = build_scenarios(seed=42, n_per_intent=3, kb=kb)
    for a, b in zip(s1, s2):
        assert a.intent == b.intent
        assert a.customer == b.customer
        assert a.facts == b.facts


def test_all_registers_used(kb):
    scenarios = build_scenarios(seed=0, n_per_intent=20, kb=kb)
    regs = {s.register for s in scenarios}
    assert regs == set(REGISTERS.keys())


# ---------------------------------------------------------------- LLM mock tests

@pytest.mark.asyncio
async def test_mock_llm_text():
    mock = MockLLM()
    text = await mock.generate("test", "say hello")
    assert "mock" in text.lower() or "northwind" in text.lower()


@pytest.mark.asyncio
async def test_mock_llm_json():
    mock = MockLLM()
    text = await mock.generate("test", 'extract claims from reply', json_mode=True)
    d = json.loads(text)
    assert isinstance(d, dict)


def test_extract_json_plain():
    assert extract_json('{"a":1}') == {"a": 1}


def test_extract_json_fenced():
    text = "```json\n{\"a\": 1}\n```"
    assert extract_json(text) == {"a": 1}


def test_extract_json_surrounded():
    text = "Here is the JSON: {\"b\": 2} and more text"
    assert extract_json(text) == {"b": 2}


# ---------------------------------------------------------------- fact check tests

def test_entity_survival_pass(kb, good_reply):
    facts = {"order_number": "39256-3341057", "price_usd": 199.00, "item": "Ridgeline -20°C Sleeping Bag"}
    checks = check_entity_survival(good_reply, facts)
    verdicts = {c.name: c.verdict for c in checks}
    assert verdicts.get("order_number") == "PASS"
    assert verdicts.get("price") == "PASS"
    assert verdicts.get("item_reference") == "PASS"


def test_entity_survival_wrong_order(kb, bad_reply):
    facts = {"order_number": "39256-3341057", "price_usd": 199.00, "item": "Ridgeline -20°C Sleeping Bag"}
    checks = check_entity_survival(bad_reply, facts)
    verdicts = {c.name: c.verdict for c in checks}
    assert verdicts.get("order_number") == "FAIL"


def test_policy_refund_window(kb):
    reply = "Your refund will appear within 5–7 business days after processing."
    facts = {"refund_window_text": "5–7 business days"}
    checks = check_policy_numbers(reply, facts, kb)
    verdicts = {c.name: c.verdict for c in checks}
    assert verdicts.get("refund_window") == "PASS"


def test_policy_wrong_refund_window(kb):
    reply = "Your refund will appear within 3–4 business days."
    facts = {"refund_window_text": "5–7 business days"}
    checks = check_policy_numbers(reply, facts, kb)
    has_fail = any(c.verdict == "FAIL" for c in checks if c.name == "refund_window")
    assert has_fail, f"Expected refund_window FAIL but got: {[(c.name, c.verdict) for c in checks]}"


def test_eligibility_warranty_void(kb):
    reply = "We'll ship you a free replacement immediately."
    facts = {"in_warranty": False}
    checks = check_eligibility_consistency(reply, facts)
    has_fail = any(c.verdict == "FAIL" for c in checks)
    assert has_fail


def test_eligibility_warranty_valid(kb):
    reply = "Good news — your 24-month warranty covers this issue. Please send a photo."
    facts = {"in_warranty": True}
    checks = check_eligibility_consistency(reply, facts)
    verdicts = [c.verdict for c in checks]
    assert "FAIL" not in verdicts


def test_gates_forbidden_phrase(kb):
    reply = "As an AI, I cannot help you. — The Northwind Team"
    checks = check_gates(reply, kb)
    has_forbidden = any(c.name == "forbidden_phrase" and c.verdict == "FAIL" for c in checks)
    has_prompt = any(c.name == "prompt_leak" and c.verdict == "FAIL" for c in checks)
    assert has_forbidden or has_prompt


def test_gates_missing_signature(kb):
    reply = "We'll process your return right away. Thanks!"
    checks = check_gates(reply, kb)
    has_sig = any(c.name == "signature" and c.verdict == "FAIL" for c in checks)
    assert has_sig


# ---------------------------------------------------------------- composite tests

def test_tone_norm():
    assert tone_norm(1) == 0.0
    assert tone_norm(5) == 1.0
    assert 0.6 < tone_norm(4) < 0.9


def test_weights_sum_to_one():
    total = sum(WEIGHTS.values())
    assert abs(total - 1.0) < 0.001, f"Weights sum to {total}, expected 1.0"


# ---------------------------------------------------------------- validate module tests

def test_spearman_perfect():
    x = [1, 2, 3, 4, 5]
    y = [10, 20, 30, 40, 50]
    assert abs(spearman(x, y) - 1.0) < 0.001


def test_spearman_inverse():
    x = [1, 2, 3, 4, 5]
    y = [50, 40, 30, 20, 10]
    assert abs(spearman(x, y) + 1.0) < 0.001


def test_pearson_perfect():
    x = [1.0, 2.0, 3.0]
    y = [2.0, 4.0, 6.0]
    assert abs(pearson(x, y) - 1.0) < 0.001


def test_perturbations_all_modify():
    """Each perturbation function actually changes the reply."""
    import yaml
    kb = yaml.safe_load((Path(__file__).parents[1] / "data" / "knowledge_base.yaml").read_text())
    reply = ("Hi! We've approved the refund for order 12345-6789012 ($199.00). "
             "Expect it within 5–7 business days. — The Northwind Team")
    facts = {"order_number": "12345-6789012", "price_usd": 199.0,
             "in_warranty": False, "adjustment_approved": False, "already_shipped": True}
    for p in PERTURBATIONS:
        corrupted = p.fn(reply, facts, kb)
        assert corrupted != reply, f"Perturbation {p.id} ({p.name}) didn't change the reply"
