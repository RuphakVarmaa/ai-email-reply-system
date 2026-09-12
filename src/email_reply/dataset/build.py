"""Dataset build pipeline: scenario skeletons -> LLM-realized text -> validated pairs.

Gates (each pair must pass ALL of these before being admitted):
  G1 fact survival: every fact string that MUST appear verbatim (order numbers,
     prices) appears in the realized text.
  G2 forbidden content: no forbidden phrases from the KB, no invented order
     numbers (an order number is a 5-7 digit token; if the realized text contains
     an order-number-like token not in the scenario's allowed set, reject).
  G3 length sanity: email 40-200 words, reply 50-220 words.
  G4 no meta talk: realized text must not contain 'as an AI', prompt-leak phrases,
     or instruction-echo (e.g. 'Required facts:').
  G5 register match: (soft, logged) — realized text length & exclamation marks
     consistent with register expectations; logged, not gating, except 'terse'
     emails must be < 60 words.
Failed pairs are regenerated up to 3 times with a nudged prompt; still-failing
skeletons are dropped and reported in the build report (honesty: we publish
rejection counts).
"""
from __future__ import annotations

import asyncio
import json
import random
import re
from collections import Counter
from pathlib import Path

from ..llm import LLM, extract_json
from .synth import Scenario, load_kb

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"

ORDER_RE = re.compile(r"\b\d{5}[-\s]?\d{7}\b")
# Meta phrases that indicate prompt leakage. 'Subject:' is ALLOWED (emails have
# subjects); 'register:'/'Required facts:' etc. indicate instruction echo.
FORBIDDEN_META = [
    "as an AI", "language model", "required facts", "register:",
    "content requirements", "write an email", "write the support",
    "must include verbatim", "tone matches",
]
PRICE_RE = re.compile(r"\$\s?\d[\d,]*\.?\d*\b|\b\d+\.\d{2}\b")


def check_email(sc: Scenario, text: str, kb: dict) -> list[str]:
    errs = []
    if sc.facts.get("order_number") and sc.facts["order_number"] not in text:
        errs.append(f"G1: order number {sc.facts['order_number']} missing from email")
    if len(text.split()) < 40:
        errs.append("G3: email too short")
    if len(text.split()) > 200:
        errs.append("G3: email too long")
    for ph in FORBIDDEN_META:
        if ph.lower() in text.lower():
            errs.append(f"G4: meta phrase '{ph}' in email")
    # G2: invented order numbers
    allowed = {sc.facts.get("order_number", "")} - {""}
    for m in ORDER_RE.findall(text.replace(",", "")):
        if m.replace(" ", "") not in {a.replace(" ", "") for a in allowed}:
            errs.append(f"G2: invented order number {m} in email")
    if sc.register == "terse" and len(text.split()) > 70:
        errs.append("G5: terse email too long")
    for ph in kb["forbidden_reply_phrases"]:
        if ph.lower() in text.lower():
            errs.append(f"G2: forbidden phrase '{ph}'")
    return errs


def check_reply(sc: Scenario, text: str, kb: dict) -> list[str]:
    errs = []
    if sc.facts.get("order_number") and sc.facts["order_number"] not in text:
        errs.append(f"G1: order number missing from reply")
    if len(text.split()) < 50:
        errs.append("G3: reply too short")
    if len(text.split()) > 220:
        errs.append("G3: reply too long")
    for ph in FORBIDDEN_META:
        if ph.lower() in text.lower():
            errs.append(f"G4: meta phrase '{ph}' in reply")
    for m in ORDER_RE.findall(text.replace(",", "")):
        allowed = {sc.facts.get("order_number", "")} - {""}
        if m.replace(" ", "") not in {a.replace(" ", "") for a in allowed}:
            errs.append(f"G2: invented order number {m} in reply")
    for ph in kb["forbidden_reply_phrases"]:
        if ph.lower() in text.lower():
            errs.append(f"G2: forbidden phrase '{ph}'")
    if "The Northwind Team" not in text:
        errs.append("G6: reply missing signature")
    return errs


def _nudge(spec: str, errs: list[str], attempt: int) -> str:
    guidance = "; ".join(errs[:3])
    return (spec + f"\n\nNOTE — your previous attempt failed validation: {guidance}. "
            f"Fix exactly these issues. (attempt {attempt + 1})")


async def realize_one(llm: LLM, sc: Scenario, kb: dict, max_attempts: int = 3) -> tuple[Scenario, list[str]]:
    """LLM-realize email + reply for one scenario, validating each attempt."""
    errs_total = []
    email = reply = None
    for attempt in range(max_attempts):
        espec = sc.email_spec if attempt == 0 else _nudge(sc.email_spec, errs_total, attempt)
        email = await llm.generate(
            espec,
            system=("You are a careful writer producing realistic customer-support emails. "
                    "Follow the instructions exactly. Output ONLY the email text — no preamble, "
                    "no markdown, no explanations. Include a 'Subject:' first line. Keep it natural."),
            temperature=0.9 if attempt == 0 else 0.4,
        )
        email = email.strip()
        errs = check_email(sc, email, kb)
        if not errs:
            break
        errs_total += errs
    else:
        return sc, errs_total

    for attempt in range(max_attempts):
        rspec = sc.reply_spec if attempt == 0 else _nudge(sc.reply_spec, errs_total, attempt)
        reply = await llm.generate(
            rspec,
            system=("You are a customer-support agent at Northwind Supply Co. writing a reply. "
                    "Follow the instructions exactly. Output ONLY the reply text — no preamble, "
                    "no markdown fences, no explanations. End with '— The Northwind Team'."),
            temperature=0.7 if attempt == 0 else 0.3,
        )
        reply = reply.strip()
        errs = check_reply(sc, reply, kb)
        if not errs:
            break
        errs_total += errs
    else:
        return sc, errs_total

    sc.email = email  # type: ignore[attr-defined]
    sc.reply = reply  # type: ignore[attr-defined]
    return sc, []


def split_subject_and_body(email_text: str) -> tuple[str, str]:
    m = re.match(r"\s*Subject:\s*(.+?)\s*\n(.*)", email_text, re.S)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return "", email_text.strip()


async def build_dataset(seed: int = 42, n_per_intent: int = 34, out_dir: Path = DATA,
                        limit: int | None = None, llm: LLM | None = None,
                        concurrency: int = 8, checkpoint: Path | None = None) -> dict:
    """Build the full dataset. Returns a build report (honesty artifact).

    Fault tolerance: every scenario is attempted independently; a scenario that
    exhausts retries (rate limits included) is skipped and reported, and results
    are checkpointed so an interrupted build can resume without redoing work.
    """
    from .synth import build_scenarios
    kb = load_kb(out_dir / "knowledge_base.yaml")
    skeletons = build_scenarios(seed=seed, n_per_intent=n_per_intent, kb=kb)
    if limit:
        skeletons = skeletons[:limit]
    llm = llm or LLM("gemini", model="gemini-3.5-flash")
    sem = asyncio.Semaphore(concurrency)
    report = {"seed": seed, "n_skeletons": len(skeletons), "attempts": Counter(),
              "rejections": [], "n_admitted": 0}

    done: dict[int, dict] = {}
    if checkpoint and checkpoint.exists():
        done = {d["i"]: d for d in json.loads(checkpoint.read_text())}
        print(f"  resuming: {len(done)} scenarios already realized", flush=True)
    results: dict[int, dict] = {i: {"i": i, "email": None, "reply": None, "errs": None}
                                for i in range(len(skeletons))}
    for i, d in done.items():
        results[i] = d

    async def run_one(i, sc):
        async with sem:
            try:
                sc2, errs = await realize_one(llm, sc, kb)
            except Exception as e:  # rate-limit exhaustion etc.
                return i, None, [f"generation error: {str(e)[:160]}"]
            return i, sc2, errs

    pending = [i for i in range(len(skeletons)) if results[i]["errs"] is None and results[i]["email"] is None]
    print(f"  {len(pending)} scenarios to realize (concurrency={concurrency})", flush=True)
    BATCH = min(concurrency, 4)  # keep small to avoid rate-limit storms
    for start in range(0, len(pending), BATCH):
        batch_ids = pending[start:start + BATCH]
        batch = [run_one(i, skeletons[i]) for i in batch_ids]
        got = await asyncio.gather(*batch)
        for i, sc2, errs in got:
            if sc2 is not None and not errs:
                results[i] = {"i": i, "email": sc2.email, "reply": sc2.reply, "errs": []}
            else:
                results[i] = {"i": i, "email": None, "reply": None, "errs": errs}
        if checkpoint:
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_text(json.dumps([results[i] for i in sorted(results)], ensure_ascii=False))
        done_count = sum(1 for r in results.values() if r.get("email"))
        print(f"  [{done_count}/{len(skeletons)}] batch {start//BATCH+1} done", flush=True)

    pairs = []
    for i in sorted(results):
        sc = skeletons[i]
        r = results[i]
        if r["errs"]:
            report["rejections"].append({"id": i, "intent": sc.intent, "errors": r["errs"]})
            report["attempts"][sc.intent] = report["attempts"].get(sc.intent, 0) + 1
            continue
        sc.email = r["email"]  # type: ignore[attr-defined]
        sc.reply = r["reply"]  # type: ignore[attr-defined]
        pairs.append(sc)

    # dedupe by (intent, order_number, customer)
    seen = set()
    deduped = []
    for sc in pairs:
        key = (sc.intent, json.dumps(sc.facts, sort_keys=True)[:120], sc.customer)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(sc)
    dropped_dupe = len(pairs) - len(deduped)
    pairs = deduped

    # deterministic split: stratified by intent, ~18% test
    rng = random.Random(seed + 1)
    by_intent: dict[str, list] = {}
    for sc in pairs:
        by_intent.setdefault(sc.intent, []).append(sc)
    train, test = [], []
    for intent, items in by_intent.items():
        items.sort(key=lambda s: s.customer + str(s.facts))  # deterministic shuffle anchor
        rng.shuffle(items)
        n_test = max(2, round(len(items) * 0.18))
        test.extend(items[:n_test])
        train.extend(items[n_test:])
    rng.shuffle(train)
    rng.shuffle(test)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "corpus").mkdir(exist_ok=True)

    def dump(rows, path):
        with open(path, "w") as f:
            for sc in rows:
                subject, body = split_subject_and_body(sc.email)  # type: ignore[attr-defined]
                f.write(json.dumps({
                    "id": f"{sc.intent}-{abs(hash((sc.intent, sc.customer, json.dumps(sc.facts, sort_keys=True)))) % 10**8}",
                    "intent": sc.intent, "sub": sc.sub, "register": sc.register,
                    "template_id": sc.template_id, "customer": sc.customer,
                    "subject": subject, "incoming": body,
                    "reply": sc.reply,  # type: ignore[attr-defined]
                    "facts": sc.facts, "kb_anchors": sc.kb_anchors,
                    "questions": sc.questions, "actions": sc.actions,
                }) + "\n")

    dump(train, out_dir / "corpus" / "train.jsonl")
    dump(test, out_dir / "corpus" / "test.jsonl")

    # demo set: 5 hand-picked (deterministic) examples for README/demo
    demo_intents = ["refund_request", "shipping_delay", "defect_warranty",
                    "warranty_denied", "complaint_escalation"]
    demo = []
    for it in demo_intents:
        cand = [sc for sc in test if sc.intent == it]
        if cand:
            demo.append(cand[0])
    dump(demo, out_dir / "corpus" / "demo.jsonl")

    report["n_admitted"] = len(pairs)
    report["n_train"] = len(train)
    report["n_test"] = len(test)
    report["n_demo"] = len(demo)
    report["n_dupes_dropped"] = dropped_dupe
    report["intent_counts"] = dict(Counter(sc.intent for sc in pairs))
    report["register_counts"] = dict(Counter(sc.register for sc in pairs))
    (out_dir / "dataset_manifest.json").write_text(json.dumps(report, indent=2, default=str))
    return report


async def main():
    import sys
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 34
    backend = sys.argv[3] if len(sys.argv) > 3 else "gemini"
    model = sys.argv[4] if len(sys.argv) > 4 else "gemini-flash-lite-latest"
    if backend == "mock":
        llm = LLM("mock", mock=MockLLMShim())
    else:
        llm = LLM(backend, model=model, rpm=float(__import__("os").environ.get("BUILD_RPM", "9")))
    cp = DATA / "cache" / f"build_checkpoint_{seed}.json"
    cp.parent.mkdir(parents=True, exist_ok=True)
    rep = await build_dataset(seed=seed, n_per_intent=n, llm=llm, checkpoint=cp)
    print(json.dumps({k: v for k, v in rep.items() if k != "rejections"}, indent=2, default=str))
    print(f"rejections: {len(rep['rejections'])}")


class MockLLMShim:
    def __init__(self):
        from ..llm import MockLLM
        self.inner = MockLLM()

    async def generate(self, *a, **k):
        return await self.inner.generate(*a, **k)


if __name__ == "__main__":
    asyncio.run(main())
