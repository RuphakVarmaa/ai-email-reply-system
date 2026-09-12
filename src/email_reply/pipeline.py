"""Main pipeline: run agent on golden eval set with multiple modes (baselines + full).

Usage:
    python -m email_reply.pipeline [--mode llm|template|nn] [--n N] [--no-judge]
"""
import asyncio
import json
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from email_reply.agent import AppleSupportAgent, SimpleRetriever
from email_reply.evaluate import run_evaluation
from email_reply.llm import LLM

DATA = Path(__file__).resolve().parents[2] / "data"
REPORTS = Path(__file__).resolve().parents[2] / "reports"


async def main():
    args = sys.argv[1:]
    mode = "template"  # default to template (no API needed)
    n = None
    use_judge = False
    judge_model = "gemini-flash-lite-latest"
    gen_model = "gemini-flash-lite-latest"
    rpm = 6.0
    
    i = 0
    while i < len(args):
        if args[i] == "--mode": mode = args[i+1]; i += 2
        elif args[i] == "--n": n = int(args[i+1]); i += 2
        elif args[i] == "--judge": use_judge = True; i += 1
        elif args[i] == "--judge-model": judge_model = args[i+1]; i += 2
        elif args[i] == "--gen-model": gen_model = args[i+1]; i += 2
        elif args[i] == "--rpm": rpm = float(args[i+1]); i += 2
        else: i += 1
    
    # Load data
    with open(DATA / "processed" / "apple_clean.jsonl") as f:
        train_pairs = [json.loads(l) for l in f]
    with open(DATA / "golden_eval" / "golden_200.jsonl") as f:
        golden = [json.loads(l) for l in f]
    if n:
        golden = golden[:n]
    
    print(f"Pipeline: mode={mode}, eval_set={len(golden)}, judge={'yes' if use_judge else 'no'}")
    print("=" * 70)
    
    # Build components
    retriever = SimpleRetriever(train_pairs)
    llm = None
    if mode == "llm":
        llm = LLM("gemini", model=gen_model, rpm=rpm)
    agent = AppleSupportAgent(retriever, llm=llm)
    
    judge_llm = None
    if use_judge:
        judge_llm = LLM("gemini", model=judge_model, rpm=rpm)
    
    t0 = time.time()
    report = await run_evaluation(golden, agent, judge_llm=judge_llm, mode=mode, max_n=n)
    elapsed = time.time() - t0
    report["elapsed_s"] = round(elapsed, 1)
    
    # Save
    REPORTS.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS / f"eval_{mode}.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n" + "=" * 70)
    print(f"MODE: {mode} | N={report['n']} | Time: {elapsed:.1f}s")
    if report.get("intent"):
        print(f"Intent accuracy: {report['intent'].get('accuracy', 'N/A')}")
    if report.get("escalation"):
        esc = report["escalation"]
        print(f"Escalation: P={esc.get('precision','?')} R={esc.get('recall','?')} F1={esc.get('f1','?')}")
    if report.get("judge"):
        j = report["judge"]
        print(f"Judge: overall={j.get('mean_overall','?')} would_send={j.get('would_send_rate','?')}")
    auto = report.get("automated", {})
    print(f"ROUGE-L: {auto.get('mean_rouge_l','?')} | Keyword overlap: {auto.get('mean_keyword_overlap','?')}")
    print(f"Report saved: {out_path}")
    
    return report


if __name__ == "__main__":
    asyncio.run(main())
