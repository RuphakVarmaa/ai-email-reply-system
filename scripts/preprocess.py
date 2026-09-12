#!/usr/bin/env python3
"""
Preprocess AppleSupport Twitter customer-support pairs.

Reads   : data/raw/apple_pairs.jsonl
Writes  : data/processed/apple_clean.jsonl
"""

import json
import os
import random
import re
from collections import OrderedDict

# ── paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RAW_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "apple_pairs.jsonl")
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
OUT_PATH = os.path.join(OUT_DIR, "apple_clean.jsonl")

# ── helpers ──────────────────────────────────────────────────────────────────
URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
MENTION_RE = re.compile(r"^(@\w+\s*)+")            # leading @user mentions
HTML_ENT = {"&gt;": ">", "&lt;": "<", "&amp;": "&"}


def clean_text(text: str) -> str:
    """Strip leading @mentions, decode HTML entities, collapse whitespace."""
    # Decode common HTML entities
    for ent, char in HTML_ENT.items():
        text = text.replace(ent, char)
    # Remove leading @username mention(s)
    text = MENTION_RE.sub("", text)
    # Collapse whitespace
    text = " ".join(text.split()).strip()
    return text


def is_only_url(text: str) -> bool:
    """True when the raw text (before cleaning) is empty or just URL(s)."""
    stripped = URL_RE.sub("", text).strip()
    # Also strip leading mentions to see if anything remains
    stripped = MENTION_RE.sub("", stripped).strip()
    return len(stripped) == 0


def is_dm_redirect(text: str) -> bool:
    """
    True when the brand reply is essentially a DM redirect:
    contains 'DM us' or a t.co link, with fewer than 15 non-URL words.
    """
    lower = text.lower()
    has_dm_cue = "dm us" in lower or "dm me" in lower
    has_tco = "https://t.co/" in lower

    if not (has_dm_cue or has_tco):
        return False

    # Count non-URL words
    without_urls = URL_RE.sub("", text)
    words = without_urls.split()
    return len(words) < 15


# ── main pipeline ────────────────────────────────────────────────────────────
def main():
    # 1. Load raw data
    raw = []
    with open(RAW_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw.append(json.loads(line))
    total_raw = len(raw)
    print(f"{'Raw pairs loaded:':<40} {total_raw:>7}")

    # 2a. Remove rows where customer_text or brand_text is empty or just a URL
    filtered = [
        r for r in raw
        if r.get("customer_text", "").strip()
        and r.get("brand_text", "").strip()
        and not is_only_url(r["customer_text"])
        and not is_only_url(r["brand_text"])
    ]
    print(f"{'After remove empty/URL-only:':<40} {len(filtered):>7}")

    # 2b. Clean texts (strip leading @mentions, decode entities, collapse ws)
    for row in filtered:
        row["customer_text"] = clean_text(row["customer_text"])
        row["brand_text"] = clean_text(row["brand_text"])

    # 2c. Remove DM-redirect brand replies
    filtered = [r for r in filtered if not is_dm_redirect(r["brand_text"])]
    print(f"{'After remove DM redirects:':<40} {len(filtered):>7}")

    # 2d. Remove rows where customer_text < 10 chars
    filtered = [r for r in filtered if len(r["customer_text"]) >= 10]
    print(f"{'After remove short customer_text (<10):':<40} {len(filtered):>7}")

    # 2e. Deduplicate by customer_text (keep first occurrence)
    seen = OrderedDict()
    for row in filtered:
        key = row["customer_text"]
        if key not in seen:
            seen[key] = row
    filtered = list(seen.values())
    print(f"{'After deduplicate by customer_text:':<40} {len(filtered):>7}")

    # 3. Sample 5000 pairs
    random.seed(42)
    if len(filtered) > 5000:
        sampled = random.sample(filtered, 5000)
    else:
        sampled = filtered
    print(f"{'After sampling (seed=42):':<40} {len(sampled):>7}")

    # 4. Save to data/processed/apple_clean.jsonl
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for row in sampled:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n✅ Saved {len(sampled)} cleaned pairs to {OUT_PATH}")


if __name__ == "__main__":
    main()
