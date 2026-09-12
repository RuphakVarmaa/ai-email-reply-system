"""Unit tests for the AppleSupport AI agent system.

Designed to run WITHOUT API calls — tests the keyword classifier,
escalation rules, retriever, evaluator, and text metrics.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from email_reply.agent import (
    classify_intent_keyword, should_escalate_rules, INTENTS,
    SimpleRetriever, generate_reply_template, generate_reply_nn,
)
from email_reply.evaluate import (
    rouge_l, length_appropriateness, keyword_overlap,
    intent_metrics, escalation_metrics,
)


# ---------------------------------------------------------------- intent tests

def test_intent_battery():
    intent, conf = classify_intent_keyword("My iPhone battery drains so fast after update")
    assert intent == "battery_power"
    assert conf > 0


def test_intent_update():
    intent, _ = classify_intent_keyword("How do I update to iOS 11.1?")
    assert intent == "software_update"


def test_intent_connectivity():
    intent, _ = classify_intent_keyword("WiFi keeps disconnecting on my MacBook")
    assert intent == "connectivity"


def test_intent_account():
    intent, _ = classify_intent_keyword("I can't sign in to my Apple ID")
    assert intent == "account_icloud"


def test_intent_hardware():
    intent, _ = classify_intent_keyword("My screen is cracked, is this covered under warranty?")
    assert intent == "hardware"


def test_intent_unknown():
    intent, conf = classify_intent_keyword("Hello there")
    assert intent == "general_inquiry"
    assert conf <= 0.5


def test_all_intents_have_keywords():
    from email_reply.agent import INTENT_KEYWORDS
    for intent in INTENTS:
        assert intent in INTENT_KEYWORDS, f"Missing keywords for {intent}"
        assert len(INTENT_KEYWORDS[intent]) >= 3


# ---------------------------------------------------------------- escalation tests

def test_escalate_hardware():
    esc, reason = should_escalate_rules("My screen is cracked", "hardware")
    assert esc is True
    assert "hardware" in reason.lower() or "physical" in reason.lower()


def test_escalate_angry():
    esc, reason = should_escalate_rules("This is the worst service, total scam!", "general_inquiry")
    assert esc is True


def test_escalate_billing():
    esc, reason = should_escalate_rules("I need a refund for this purchase", "app_issue")
    assert esc is True
    assert "billing" in reason.lower() or "financial" in reason.lower()


def test_no_escalate_simple():
    esc, _ = should_escalate_rules("How do I update my iPhone?", "software_update")
    assert esc is False


def test_escalate_repeated_failure():
    esc, _ = should_escalate_rules("I already tried restarting, nothing works!", "performance")
    assert esc is True


# ---------------------------------------------------------------- template reply tests

def test_template_reply_exists_for_all_intents():
    for intent in INTENTS:
        reply = generate_reply_template(intent)
        assert len(reply) > 20, f"Template for {intent} is too short"
        assert len(reply) < 600, f"Template for {intent} is too long"


def test_nn_reply_fallback():
    reply = generate_reply_nn([])
    assert len(reply) > 10


def test_nn_reply_returns_first():
    examples = [{"brand_text": "Try restarting your device.", "customer_text": "help"}]
    reply = generate_reply_nn(examples)
    assert reply == "Try restarting your device."


# ---------------------------------------------------------------- retriever tests

def test_retriever_basic():
    pairs = [
        {"customer_text": "My iPhone battery drains fast", "brand_text": "Try checking battery usage in Settings."},
        {"customer_text": "WiFi not connecting", "brand_text": "Try toggling WiFi off and on."},
        {"customer_text": "Can't update iOS", "brand_text": "Go to Settings > General > Software Update."},
    ]
    ret = SimpleRetriever(pairs)
    results = ret.search("battery dying quickly", k=2)
    assert len(results) == 2
    assert "battery" in results[0]["customer_text"].lower()


def test_retriever_empty_query():
    pairs = [{"customer_text": "test", "brand_text": "reply"}]
    ret = SimpleRetriever(pairs)
    results = ret.search("", k=1)
    assert len(results) == 1


# ---------------------------------------------------------------- evaluation metric tests

def test_rouge_l_identical():
    assert abs(rouge_l("hello world", "hello world") - 1.0) < 0.01


def test_rouge_l_no_overlap():
    assert rouge_l("hello world", "foo bar") == 0.0


def test_rouge_l_partial():
    score = rouge_l("the cat sat on the mat", "the cat on the mat")
    assert 0.5 < score < 1.0


def test_length_short():
    assert length_appropriateness("Short reply") == 1.0


def test_length_long():
    assert length_appropriateness("x" * 900) < 0.5


def test_keyword_overlap_full():
    assert keyword_overlap("battery drain iphone", "your iphone battery drain is normal") > 0.5


def test_keyword_overlap_none():
    assert keyword_overlap("battery drain", "hello world") == 0.0


def test_intent_metrics_perfect():
    preds = ["a", "b", "a"]
    labels = ["a", "b", "a"]
    m = intent_metrics(preds, labels)
    assert m["accuracy"] == 1.0


def test_intent_metrics_half():
    preds = ["a", "b"]
    labels = ["a", "a"]
    m = intent_metrics(preds, labels)
    assert m["accuracy"] == 0.5


def test_escalation_metrics():
    preds = [True, True, False, False]
    labels = [True, False, True, False]
    m = escalation_metrics(preds, labels)
    assert m["tp"] == 1
    assert m["fp"] == 1
    assert m["fn"] == 1
    assert m["tn"] == 1
    assert m["precision"] == 0.5
    assert m["recall"] == 0.5
