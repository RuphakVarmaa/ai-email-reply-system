"""Entry point for `python -m email_reply`."""
import asyncio
import sys


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    sys.argv = [sys.argv[0]] + sys.argv[2:]  # shift the subcommand out

    if cmd == "build":
        from .dataset.build import main as build_main
        asyncio.run(build_main())
    elif cmd == "run":
        from .run import main as run_main
        asyncio.run(run_main())
    elif cmd == "demo":
        asyncio.run(demo())
    else:
        print("Usage: python -m email_reply <build|run|demo>")
        print("  build   Build the synthetic dataset")
        print("  run     Generate replies + evaluate (full pipeline)")
        print("  demo    Run a single demo email through the pipeline")
        sys.exit(0 if cmd == "help" else 1)


async def demo():
    import json
    from pathlib import Path
    from .gen.generate import build_generator
    from .eval.facts import run_fact_checks
    from .eval.judge import Judge
    from .eval.composite import score_response
    from .llm import LLM
    import yaml

    DATA = Path(__file__).resolve().parents[1] / "data"
    kb = yaml.safe_load((DATA / "knowledge_base.yaml").read_text())
    demo_path = DATA / "corpus" / "demo.jsonl"
    if not demo_path.exists():
        print("Demo set not found. Run 'python -m email_reply build' first.")
        return

    with open(demo_path) as f:
        pair = json.loads(f.readline())

    print("="*70)
    print("INCOMING EMAIL")
    print(f"  Subject: {pair['subject']}")
    print(f"  {pair['incoming'][:500]}")
    print("="*70)

    generator = await build_generator(rpm=4)
    gen = await generator.generate(
        incoming=pair["incoming"], subject=pair.get("subject", ""),
        mode="rag", facts_hint=pair.get("facts"),
    )

    print("\nGENERATED REPLY")
    print(gen.reply)
    print("\n" + "="*70)

    judge_llm = LLM("gemini", model="gemini-flash-lite-latest", rpm=4)
    judge = Judge(judge_llm)
    jr = await judge.judge(
        incoming=pair["incoming"], subject=pair.get("subject", ""),
        reply=gen.reply, facts=pair["facts"],
        questions=pair.get("questions", []),
        actions=pair.get("actions", []),
        register=pair.get("register", "calm"),
    )
    rs = score_response(jr, pair["facts"], kb, gen.reply)
    rs.id = pair["id"]
    rs.intent = pair["intent"]

    print("EVALUATION")
    print(f"  Composite score: {rs.composite:.4f}")
    print(f"  Components: {json.dumps(rs.components, indent=4)}")
    if rs.gate_failures:
        print(f"  ⚠️  Gate failures: {rs.gate_failures}")
    print(f"  Fact checks ({len(rs.fact_checks)}):")
    for fc in rs.fact_checks:
        print(f"    [{fc['verdict']}] {fc['name']}: {fc['detail']}")
    print(f"  Claims ({len(rs.claims)}):")
    for v in rs.verdicts:
        print(f"    [{v.get('verdict','?')}] {v.get('claim','?')}: {v.get('why','?')}")
    print("="*70)


if __name__ == "__main__":
    main()
