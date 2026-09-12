"""Core pipeline: intent classification + reply generation + escalation routing.

Built for AppleSupport Twitter conversations.
Uses Gemini API for LLM calls with rate limiting.
"""
from __future__ import annotations

import asyncio
import json
import re
import random
import hashlib
from pathlib import Path
from dataclasses import dataclass, field

DATA = Path(__file__).resolve().parents[2] / "data"

# ============================================================
# INTENT TAXONOMY — derived from data analysis
# ============================================================

INTENTS = {
    "software_update": "Issues with iOS/macOS/watchOS updates — installation failures, bugs after update, when next update",
    "battery_power": "Battery drain, charging issues, power problems, auto-shutdown",
    "performance": "Device slow, freezing, crashing, lagging, overheating",
    "app_issue": "App Store problems, specific app crashes, downloads failing, subscriptions",
    "connectivity": "WiFi, Bluetooth, cellular, hotspot, AirDrop not working",
    "screen_display": "Screen frozen, touch not responding, display glitches, brightness issues",
    "account_icloud": "Apple ID, iCloud, password, sign-in, two-factor, storage",
    "audio_media": "Apple Music, speakers, microphone, headphones, AirPods issues",
    "hardware": "Physical damage, warranty, repair, device not turning on, hardware defect",
    "general_inquiry": "How-to questions, feature requests, general praise/complaint, other",
}

INTENT_KEYWORDS = {
    "software_update": ["update", "upgrade", "ios ", "ios11", "macos", "high sierra", "watchos", "latest version", "11.1", "11.0", "11.2", "beta"],
    "battery_power": ["battery", "charging", "charge", "power", "drain", "draining", "dies", "dead", "plug", "won't turn on", "auto shutdown"],
    "performance": ["slow", "freeze", "frozen", "crash", "lag", "stuck", "hang", "overheat", "hot", "glitch", "bug", "unresponsive"],
    "app_issue": ["app store", "download", "can't download", "subscription", "purchase", "app crash", "app won't", "apps"],
    "connectivity": ["wifi", "wi-fi", "bluetooth", "cellular", "hotspot", "airdrop", "connect", "network", "signal"],
    "screen_display": ["screen", "display", "touch", "brightness", "black screen", "flicker", "rotation"],
    "account_icloud": ["apple id", "icloud", "password", "sign in", "sign out", "login", "two factor", "verification", "storage", "locked out"],
    "audio_media": ["music", "apple music", "speaker", "microphone", "airpod", "headphone", "sound", "volume", "audio"],
    "hardware": ["warranty", "repair", "genius", "cracked", "broken", "hardware", "defect", "replacement", "genius bar"],
    "general_inquiry": ["how do", "how to", "question", "help", "thanks", "love", "hate", "when will", "feature"],
}

# ============================================================
# ESCALATION RULES
# ============================================================

ESCALATION_RULES = {
    "auto_handle": [
        "Simple how-to questions with clear answers",
        "Standard troubleshooting (restart, update, reset settings)",
        "Status inquiries about known issues",
        "General praise/thanks",
    ],
    "escalate_to_human": [
        "Account security issues (locked out, hacked, unauthorized charges)",
        "Hardware defects requiring physical inspection or repair",
        "Billing/refund disputes",
        "Customer expressing extreme frustration (profanity, threats)",
        "Multi-step troubleshooting that failed",
        "Privacy/data loss concerns",
        "Issues requiring access to account-specific data",
    ],
}

ESCALATION_KEYWORDS_HUMAN = [
    "hacked", "stolen", "unauthorized", "charge", "refund", "money",
    "lawyer", "lawsuit", "sue", "legal", "BBB", "FTC",
    "broken", "cracked", "warranty", "repair", "genius bar",
    "lost all", "data loss", "wiped", "erased everything",
    "fucking", "shit", "bullshit", "worst", "terrible", "scam",
    "already tried", "nothing works", "been trying for", "3rd time",
    "multiple times", "hours",
]

# ============================================================
# KEYWORD-BASED INTENT CLASSIFIER (baseline)
# ============================================================

def classify_intent_keyword(text: str) -> tuple[str, float]:
    """Trivial baseline: keyword matching."""
    text_l = text.lower()
    scores = {}
    for intent, keywords in INTENT_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_l)
        if score > 0:
            scores[intent] = score
    if not scores:
        return "general_inquiry", 0.3
    best = max(scores, key=scores.get)
    confidence = min(scores[best] / 3.0, 1.0)
    return best, round(confidence, 2)


# ============================================================
# ESCALATION ROUTER (rule-based baseline)
# ============================================================

def should_escalate_rules(text: str, intent: str) -> tuple[bool, str]:
    """Rule-based escalation. Returns (escalate, reason)."""
    text_l = text.lower()
    # Always escalate hardware/account issues
    if intent == "hardware":
        return True, "Hardware issue likely requires physical inspection or warranty check"
    if intent == "account_icloud" and any(kw in text_l for kw in ["locked", "hacked", "stolen", "unauthorized"]):
        return True, "Account security concern requires human verification"
    # Escalate angry customers
    anger_words = ["fucking", "shit", "bullshit", "worst", "terrible", "scam", "lawsuit", "sue"]
    if sum(1 for w in anger_words if w in text_l) >= 1:
        return True, "Customer expressing strong frustration — needs human empathy"
    # Escalate repeated failures
    if any(p in text_l for p in ["already tried", "nothing works", "tried everything", "3rd time", "multiple times"]):
        return True, "Customer reports repeated failed troubleshooting — needs escalation"
    # Escalate billing
    if any(kw in text_l for kw in ["refund", "charge", "billing", "money", "paid"]):
        return True, "Financial/billing issue requires authorized human agent"
    return False, "Standard issue — can be auto-handled with troubleshooting guidance"


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class AgentResponse:
    customer_text: str
    intent: str
    intent_confidence: float
    intent_method: str
    reply: str
    reply_method: str
    escalate: bool
    escalate_reason: str
    retrieved_examples: list = field(default_factory=list)


# ============================================================
# LLM HELPERS (reuse from llm.py)
# ============================================================

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from email_reply.llm import LLM, extract_json


# ============================================================
# LLM-BASED INTENT CLASSIFIER
# ============================================================

INTENT_CLASSIFY_PROMPT = """Classify this Apple Support customer tweet into exactly ONE intent.

Intents:
{intents}

Customer tweet: "{text}"

Return JSON: {{"intent": "<intent_id>", "confidence": <0.0-1.0>, "reasoning": "one sentence"}}"""

async def classify_intent_llm(llm: LLM, text: str) -> tuple[str, float, str]:
    intents_str = "\n".join(f"- {k}: {v}" for k, v in INTENTS.items())
    prompt = INTENT_CLASSIFY_PROMPT.format(intents=intents_str, text=text[:500])
    raw = await llm.generate(prompt, temperature=0.0, json_mode=True)
    d = extract_json(raw)
    intent = d.get("intent", "general_inquiry")
    if intent not in INTENTS:
        intent = "general_inquiry"
    return intent, float(d.get("confidence", 0.5)), d.get("reasoning", "")


# ============================================================
# LLM-BASED ESCALATION ROUTER
# ============================================================

ESCALATION_PROMPT = """You are an AppleSupport triage agent. Decide if this customer message should be:
- AUTO-HANDLED: standard troubleshooting, how-to, or simple issue the bot can resolve
- ESCALATED TO HUMAN: needs account access, physical repair, billing dispute, angry customer needing empathy, complex multi-step failure, or privacy/security concern

Customer tweet: "{text}"
Detected intent: {intent}

Return JSON: {{"escalate": true/false, "reason": "one sentence explaining why", "urgency": "low|medium|high"}}"""

async def should_escalate_llm(llm: LLM, text: str, intent: str) -> tuple[bool, str]:
    prompt = ESCALATION_PROMPT.format(text=text[:500], intent=intent)
    raw = await llm.generate(prompt, temperature=0.0, json_mode=True)
    d = extract_json(raw)
    return bool(d.get("escalate", False)), d.get("reason", "no reason given")


# ============================================================
# RETRIEVAL: find similar past conversations
# ============================================================

def _tokenize(text: str) -> list[str]:
    return [w for w in re.findall(r'[a-z0-9]+', text.lower()) if len(w) > 2]


class SimpleRetriever:
    """TF-IDF cosine retriever over past conversations. Pure Python, no deps."""
    
    def __init__(self, pairs: list[dict]):
        self.pairs = pairs
        self.docs = [_tokenize(p["customer_text"]) for p in pairs]
        # Build IDF
        from collections import Counter
        import math
        self.df = Counter()
        for doc in self.docs:
            self.df.update(set(doc))
        self.N = len(pairs)
        self._idf_cache = {}
    
    def _idf(self, term):
        if term not in self._idf_cache:
            import math
            self._idf_cache[term] = math.log((self.N + 1) / (self.df.get(term, 0) + 1)) + 1
        return self._idf_cache[term]
    
    def _tfidf(self, tokens):
        from collections import Counter
        tf = Counter(tokens)
        n = max(len(tokens), 1)
        return {t: (c/n) * self._idf(t) for t, c in tf.items()}
    
    def search(self, query: str, k: int = 5, intent_filter: str | None = None) -> list[dict]:
        import math
        q_vec = self._tfidf(_tokenize(query))
        scores = []
        for i, doc in enumerate(self.docs):
            if intent_filter and self.pairs[i].get("intent") != intent_filter:
                continue
            d_vec = self._tfidf(doc)
            num = sum(w * d_vec.get(t, 0) for t, w in q_vec.items())
            den_q = math.sqrt(sum(w*w for w in q_vec.values())) or 1
            den_d = math.sqrt(sum(w*w for w in d_vec.values())) or 1
            scores.append((num / (den_q * den_d), i))
        scores.sort(reverse=True)
        return [self.pairs[i] for _, i in scores[:k]]


# ============================================================
# REPLY GENERATOR
# ============================================================

REPLY_PROMPT = """You are @AppleSupport on Twitter, replying to a customer.

Brand voice: helpful, concise (Twitter's 280 char spirit — aim for 1-3 sentences),
technical but accessible, empathetic, never defensive.

Here are similar past conversations for reference:
{examples}

Customer tweet: "{text}"
Detected intent: {intent}

Rules:
1. Address the specific issue, don't give generic advice
2. If you can solve it, give the exact steps (Settings > General > ...)
3. If it needs more info, ask ONE specific diagnostic question
4. If it needs a DM/call, say so and why
5. Never say "I'm an AI" or mention being automated
6. Keep it under 280 characters if possible, max 2 tweets worth
7. Start with the customer's concern, not "Hi" or "Hello"

Write the reply (plain text, no quotes, no preamble):"""

NEAREST_NEIGHBOR_REPLY = """[Nearest-neighbor baseline: returning most similar historical reply]"""


async def generate_reply_llm(llm: LLM, text: str, intent: str,
                              examples: list[dict]) -> str:
    ex_str = ""
    for i, ex in enumerate(examples[:3], 1):
        ex_str += f"Example {i}:\n  Customer: {ex['customer_text'][:150]}\n  Apple: {ex['brand_text'][:150]}\n\n"
    prompt = REPLY_PROMPT.format(examples=ex_str, text=text[:500], intent=intent)
    reply = await llm.generate(prompt, temperature=0.4, max_tokens=300)
    return reply.strip()


def generate_reply_nn(examples: list[dict]) -> str:
    """Trivial baseline: return the brand reply of the most similar past conversation."""
    if examples:
        return examples[0]["brand_text"]
    return "We'd like to help! Can you tell us more about what's happening? ^AB"


def generate_reply_template(intent: str) -> str:
    """Simple baseline: template reply by intent."""
    templates = {
        "software_update": "We'd like to help with your update issue. Try restarting your device, then go to Settings > General > Software Update. If the issue persists, let us know your device model and current iOS version.",
        "battery_power": "Sorry to hear about the battery issue. Try Settings > Battery to check usage. Also, make sure you're on the latest iOS. If the drain continues, we can look deeper — what device and iOS version?",
        "performance": "We understand the frustration with performance issues. A restart often helps: hold the power button, slide to power off, wait 30 seconds. If it continues, try Settings > General > Reset > Reset All Settings (this won't delete data).",
        "app_issue": "We'd like to help with that app issue. Try force-closing the app (double-tap Home, swipe up), then reopen. If that doesn't help, try deleting and reinstalling the app. What app is affected?",
        "connectivity": "Let's get your connection sorted. Try toggling WiFi/Bluetooth off and on in Settings. If that doesn't help, try Settings > General > Reset > Reset Network Settings. Which connection type is affected?",
        "screen_display": "We want to help with your screen issue. Try a force restart: hold Power + Home (or Volume Down on iPhone 7+) for 10 seconds. If the display issue persists, what exactly are you seeing?",
        "account_icloud": "We'd like to help with your account. For security, we recommend continuing via DM. Please DM us with the email associated with your Apple ID and we'll assist from there.",
        "audio_media": "Sorry about the audio issue. Check Settings > Sounds to make sure volume is up. Try connecting/disconnecting headphones. If using Bluetooth, forget the device and re-pair. What device are you using?",
        "hardware": "We understand your concern about the hardware. For the best assistance, we'd recommend visiting an Apple Store or Authorized Service Provider. You can also contact us via DM for further options.",
        "general_inquiry": "Thanks for reaching out! We'd be happy to help. Could you give us a bit more detail about what you need assistance with?",
    }
    return templates.get(intent, templates["general_inquiry"])


# ============================================================
# FULL AGENT PIPELINE
# ============================================================

class AppleSupportAgent:
    def __init__(self, retriever: SimpleRetriever, llm: LLM | None = None):
        self.retriever = retriever
        self.llm = llm
    
    async def process(self, customer_text: str, mode: str = "llm") -> AgentResponse:
        # 1. Intent classification
        if mode == "llm" and self.llm:
            intent, conf, _ = await classify_intent_llm(self.llm, customer_text)
            intent_method = "llm"
        else:
            intent, conf = classify_intent_keyword(customer_text)
            intent_method = "keyword"
        
        # 2. Retrieval
        examples = self.retriever.search(customer_text, k=5)
        
        # 3. Reply generation
        if mode == "llm" and self.llm:
            reply = await generate_reply_llm(self.llm, customer_text, intent, examples)
            reply_method = "llm_rag"
        elif mode == "template":
            reply = generate_reply_template(intent)
            reply_method = "template"
        elif mode == "nn":
            reply = generate_reply_nn(examples)
            reply_method = "nearest_neighbor"
        else:
            reply = generate_reply_template(intent)
            reply_method = "template"
        
        # 4. Escalation
        if mode == "llm" and self.llm:
            escalate, esc_reason = await should_escalate_llm(self.llm, customer_text, intent)
        else:
            escalate, esc_reason = should_escalate_rules(customer_text, intent)
        
        return AgentResponse(
            customer_text=customer_text,
            intent=intent, intent_confidence=conf, intent_method=intent_method,
            reply=reply, reply_method=reply_method,
            escalate=escalate, escalate_reason=esc_reason,
            retrieved_examples=[(e["customer_text"][:100], e["brand_text"][:100]) for e in examples[:3]],
        )
