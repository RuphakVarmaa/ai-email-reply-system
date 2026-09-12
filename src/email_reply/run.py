"""End-to-end runner: generate replies for the test set and evaluate them.

Usage:
    python -m email_reply.run [--mode rag|zero_shot|no_kb|nn] [--model MODEL]
                               [--judge-model MODEL] [--gen-backend gemini]
                               [--n N]  # limit to N test examples
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import yaml

from .gen.generate import ReplyGenerator, load_kb, build_generator
from .gen.retrieve import HybridRetriever, load_pairs
from .eval.facts import run_fact_checks
from .eval.judge import Judge
from .eval.composite import score_response, overall_scores, ResponseScore
from .llm import LLM

DATA = Path(__file__).resolve().parents[2] / "data"
REPORTS = Path(__file__).resolve().parents[2] / "reports"


async def run_pipeline(
    mode: str = "rag",
    gen_backend: str = "gemini",
    gen_model: str = "gemini-flash-lite-latest",
    judge_model: str = "gemini-flash-lite-latest",
    n: int | None = None,
    gen_rpm: float = 6,
    judge_rpm: float = 6,
) -> dict:
    kb = load_kb()
    train_path = DATA / "corpus" / "train.jsonl"
    test_path = DATA / "corpus" / "test.jsonl"

    if not train_path.exists() or not test_path.exists():
        print("ERROR: dataset not built yet. Run: python -m email_reply.dataset.build")
        return {"error": "dataset not found"}

    # Build generator
    gen_llm = LLM(gen_backend, model=gen_model, rpm=gen_rpm)
    train_pairs = load_pairs(train_path)
    retriever = HybridRetriever(train_pairs, backend=gen_backend)
    generator = ReplyGenerator(retriever, kb, gen_llm)

    # Build judge
    judge_llm = LLM("gemini", model=judge_model, rpm=judge_rpm)
    judge = Judge(judge_llm)

    # Load test set
    test_pairs = load_pairs(test_path)
    if n:
        test_pairs = test_pairs[:n]

    print(f"Pipeline: mode={mode}, gen={gen_model}, judge={judge_model}, "
          f"test_set={len(test_pairs)} pairs")
    print("="*70)

    scores: list[ResponseScore] = []
    t0 = time.time()

    for idx, pair in enumerate(test_pairs):
        print(f"  [{idx+1}/{len(test_pairs)}] {pair['intent']:25s} ", end="", flush=True)

        # 1. generate
        gen = await generator.generate(
            incoming=pair["incoming"], subject=pair.get("subject", ""),
            mode=mode, facts_hint=pair.get("facts"),
        )

        # 2. judge
        jr = await judge.judge(
            incoming=pair["incoming"], subject=pair.get("subject", ""),
            reply=gen.reply, facts=pair["facts"],
            questions=pair.get("questions", []),
            actions=pair.get("actions", []),
            register=pair.get("register", "calm"),
        )

        # 3. composite
        rs = score_response(jr, pair["facts"], kb, gen.reply)
        rs.id = pair["id"]
        rs.intent = pair["intent"]
        rs.register = pair.get("register", "")
        scores.append(rs)

        gf = " GATE-FAIL" if rs.gate_failures else ""
        print(f"composite={rs.composite:.3f}{gf} | cov={rs.components['coverage']:.2f} "
              f"faith={rs.components['faithfulness']:.2f} "
              f"act={rs.components['action_correctness']:.2f} "
              f"tone={rs.components['tone']:.2f}")

    elapsed = time.time() - t0
    overall = overall_scores(scores)
    overall["mode"] = mode
    overall["gen_model"] = gen_model
    overall["judge_model"] = judge_model
    overall["elapsed_s"] = round(elapsed, 1)

    # Save reports
    REPORTS.mkdir(parents=True, exist_ok=True)
    per_response = []
    for s in scores:
        per_response.append({
            "id": s.id, "intent": s.intent, "register": s.register,
            "composite": s.composite,
            "components": s.components,
            "gate_failures": s.gate_failures,
            "fact_checks": s.fact_checks,
            "claims": s.claims,
            "verdicts": s.verdicts,
            "coverage_detail": s.judge_coverage_detail,
            "reply_excerpt": s.reply[:200],
        })

    report = {"overall": overall, "per_response": per_response}
    out_path = REPORTS / f"report_{mode}.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print("\n" + "="*70)
    print(f"OVERALL ({mode}):")
    print(f"  Mean composite: {overall['mean_composite']:.4f}  "
          f"95% CI: [{overall['ci95'][0]:.4f}, {overall['ci95'][1]:.4f}]")
    print(f"  Gate failure rate: {overall['gate_failure_rate']:.2%}")
    print(f"  Mean components: {json.dumps(overall['mean_components'], indent=4)}")
    print(f"  By intent: {json.dumps(overall['by_intent'], indent=4)}")
    print(f"  Score histogram [0,.2,.4,.6,.8,1]: {overall['score_histogram']}")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"  Report saved: {out_path}")

    # Also write a human-readable markdown report
    md = generate_markdown_report(overall, per_response, mode)
    md_path = REPORTS / f"report_{mode}.md"
    md_path.write_text(md)
    print(f"  Markdown report: {md_path}")

    return report


def generate_markdown_report(overall: dict, per_response: list, mode: str) -> str:
    lines = [f"# Evaluation Report — mode: `{mode}`\n"]
    lines.append(f"**Test set size**: {overall['n']}  ")
    lines.append(f"**Mean composite**: {overall['mean_composite']:.4f} "
                 f"(95% CI: [{overall['ci95'][0]:.4f}, {overall['ci95'][1]:.4f}])  ")
    lines.append(f"**Gate failure rate**: {overall['gate_failure_rate']:.2%}  ")
    lines.append(f"**Elapsed**: {overall.get('elapsed_s', '?')}s\n")
    lines.append("## Mean Components\n")
    lines.append("| Metric | Weight | Mean |")
    lines.append("|--------|--------|------|")
    weights = {"coverage": 0.30, "faithfulness": 0.25, "action_correctness": 0.20,
               "tone": 0.15, "entity_accuracy": 0.10}
    for k, w in weights.items():
        v = overall['mean_components'].get(k, 0)
        lines.append(f"| {k} | {w:.2f} | {v:.4f} |")
    lines.append("")
    lines.append("## By Intent\n")
    lines.append("| Intent | Mean Composite |")
    lines.append("|--------|---------------|")
    for k, v in sorted(overall.get('by_intent', {}).items()):
        lines.append(f"| {k} | {v:.4f} |")
    lines.append("")
    lines.append("## Score Distribution\n")
    lines.append(f"Histogram [0, .2, .4, .6, .8, 1.0]: {overall.get('score_histogram', [])}  ")
    lines.append(f"Below 0.50: {overall.get('n_below_050', '?')}  ")
    lines.append(f"Above 0.80: {overall.get('n_above_080', '?')}\n")
    lines.append("## Per-Response Scores (top 5 + bottom 5)\n")
    sorted_pr = sorted(per_response, key=lambda x: x["composite"])
    for label, items in [("Bottom 5", sorted_pr[:5]), ("Top 5", sorted_pr[-5:])]:
        lines.append(f"### {label}\n")
        for p in items:
            lines.append(f"- **{p['id']}** ({p['intent']}, {p['register']}): "
                         f"composite={p['composite']:.4f}  "
                         f"cov={p['components']['coverage']:.2f} "
                         f"faith={p['components']['faithfulness']:.2f} "
                         f"act={p['components']['action_correctness']:.2f} "
                         f"tone={p['components']['tone']:.2f}")
            if p['gate_failures']:
                lines.append(f"  ⚠️ Gates: {'; '.join(p['gate_failures'][:2])}")
        lines.append("")
    return "\n".join(lines)


async def main():
    import sys
    args = sys.argv[1:]
    mode = "rag"
    gen_model = "gemini-flash-lite-latest"
    judge_model = "gemini-flash-lite-latest"
    n = None
    gen_rpm = 6.0
    judge_rpm = 6.0
    i = 0
    while i < len(args):
        if args[i] == "--mode":
            mode = args[i + 1]; i += 2
        elif args[i] == "--model":
            gen_model = args[i + 1]; i += 2
        elif args[i] == "--judge-model":
            judge_model = args[i + 1]; i += 2
        elif args[i] == "--n":
            n = int(args[i + 1]); i += 2
        elif args[i] == "--gen-rpm":
            gen_rpm = float(args[i + 1]); i += 2
        elif args[i] == "--judge-rpm":
            judge_rpm = float(args[i + 1]); i += 2
        else:
            i += 1
    await run_pipeline(mode=mode, gen_model=gen_model, judge_model=judge_model,
                       n=n, gen_rpm=gen_rpm, judge_rpm=judge_rpm)


if __name__ == "__main__":
    asyncio.run(main())
