# AI Email Suggested-Response System

> A RAG-grounded email reply generator with a multi-metric evaluation system that
> measures what "accurate" actually means for suggested replies — and proves it.

[![Tests](https://img.shields.io/badge/tests-23%2F23%20pass-brightgreen)](#tests)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](#setup)
[![License](https://img.shields.io/badge/license-MIT-green)](#license)

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Architecture Overview](#architecture-overview)
3. [The Dataset](#the-dataset)
4. [Response Generator](#response-generator)
5. [Accuracy System — The Core](#accuracy-system--the-core)
6. [Metric Validation](#metric-validation)
7. [Results](#results)
8. [Trade-offs & Honest Limitations](#trade-offs--honest-limitations)
9. [How AI Tools Were Used](#how-ai-tools-were-used)
10. [Repo Layout](#repo-layout)

---

## Quick Start

```bash
# 1. Clone and set up
git clone https://github.com/RuphakVarmaa/ai-email-reply-system.git
cd ai-email-reply-system
make setup                    # creates .venv, installs deps

# 2. Set API keys
export GEMINI_API_KEY="your-key"     # required for LLM calls
# export OPENAI_API_KEY="your-key"   # optional, for cross-family judge

# 3. Build the dataset (or use the pre-built one in data/corpus/)
make build-dataset            # ~90 min on free-tier; dataset ships pre-built

# 4. Run the full pipeline (generate + evaluate)
make run                      # generates replies for test set + evaluates

# 5. Quick demo (single email → reply → score)
make demo

# 6. Run tests (zero API calls)
make test                     # 23 tests, <1s

# 7. Ablation study
make ablate                   # compares RAG vs zero-shot vs no-KB vs nearest-neighbor
```

### Minimal run (5 test examples)

```bash
make run-small
```

---

## Architecture Overview

```
                    ┌─────────────────────────────┐
                    │     Knowledge Base (KB)       │
                    │  policies · catalog · rules   │
                    └────────────┬──────────────────┘
                                 │
  incoming email ──► Retriever ──┤──► Retrieved past pairs (top-k)
                     (hybrid:    │
                      semantic   │
                      + lexical) │
                                 ▼
                    ┌─────────────────────────────┐
                    │   RAG Prompt Construction     │
                    │  system prompt + KB facts     │
                    │  + few-shot retrieved pairs   │
                    │  + incoming email              │
                    └────────────┬──────────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────────┐
                    │      LLM Generation          │
                    │  (Gemini 3.5-Flash/3.6-Flash) │
                    └────────────┬──────────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────────┐
                    │    Multi-Metric Evaluation    │
                    │                               │
                    │  ┌─ Programmatic Fact Checks  │
                    │  │  (entity, policy, decision) │
                    │  │                             │
                    │  ├─ LLM Judge (CoT, rubric)    │
                    │  │  (coverage, faithfulness,   │
                    │  │   tone, action correctness) │
                    │  │                             │
                    │  └─ Gates (safety, forbidden)  │
                    │                               │
                    │  ──► Composite Score + Evidence │
                    └─────────────────────────────┘
```

---

## The Dataset

### Source & honesty

The dataset is **fully synthetic**, built from structured scenario templates
grounded in a **fictional company knowledge base** (Northwind Supply Co., an
outdoor-gear retailer). We considered real corpora:

| Option | Verdict | Why |
|--------|---------|-----|
| Enron (2001 corporate email) | Rejected as primary | No ground-truth facts → can't validate factual accuracy of replies; reply chains sparse; 2001-era tone |
| Public support corpora | Rejected | Tiny, toy, same no-facts problem |
| **Synthetic from fact templates** | **Chosen** | Full ground truth per pair; controllable difficulty; reproducible from a seed |

### Why synthetic is *more honest* here

The challenge asks us to *measure how accurate a reply is*. For that, we need
to *know* the right answer — the correct refund window, the right policy, whether
the warranty applies. Real email corpora don't have machine-checkable ground truth.
Our synthetic dataset does: every pair ships with a `facts` dict containing the
exact order number, price, policy rules, and correct decision. The evaluator checks
the generated reply against these facts **programmatically** — no LLM can fool a
regex looking for the right order number.

### Construction pipeline

1. **Scenario skeletons** (deterministic, seed=42): 12 intent families × ~34 each = 408 scenarios
   - Intents: refund_request, shipping_delay, defect_warranty, exchange_size, order_cancel,
     order_modify, product_question, warranty_denied, price_adjustment, shipping_options,
     gift_card, complaint_escalation
   - Each skeleton defines: customer name, register (calm/frustrated/furious/terse/chatty/
     professional/grateful), ground-truth facts, KB anchors, expected questions & actions

2. **LLM realization** (Gemini Flash Lite): each skeleton's structured spec is sent to an LLM
   that writes natural email + reply text, following strict instructions (word count, required
   facts, tone, forbidden content)

3. **Programmatic validation gates**: every realized pair must pass:
   - G1 Fact survival: required facts appear verbatim
   - G2 No invented entities: no order numbers not in the scenario
   - G3 Length sanity: email 40–200 words, reply 50–220 words
   - G4 No meta talk: no "as an AI", no instruction echo
   - G5 No forbidden phrases from KB
   - G6 Signature present
   - Failures trigger up to 3 re-generations with a nudged prompt

4. **Split**: ~82% train (RAG corpus) / ~18% test + 5 demo examples

### Honest limitations

- Synthetic emails are cleaner than real mail (no forwarding chains, no signature blocks)
- Intent mix is ours to choose (but documented and balanced)
- LLM-realized text may carry phrasing tics; we mitigate with register variety
- The KB is small but complete for this world
- All seeds and templates are published for full reproducibility

---

## Response Generator

### Design: RAG with KB grounding

Given a new incoming email:
1. **Hybrid retrieval** over past (incoming, reply) pairs:
   - Semantic: Gemini Embedding 001 (256-dim) → cosine similarity
   - Lexical: TF-IDF token overlap (catches order numbers, SKUs)
   - Combined: 0.5 × semantic + 0.5 × lexical
2. **Top-k pairs** (k=6 default) become few-shot examples in the prompt
3. **KB facts** relevant to the retrieved scenarios are injected explicitly
4. **System prompt** enforces the brand voice and grounding rules
5. **LLM generation** (Gemini Flash, temperature=0.4)

### Why RAG over alternatives

| Approach | Pros | Cons | Our choice |
|----------|------|------|------------|
| Fine-tuning | Captures brand voice deeply | 350 pairs is marginal; facts baked in (stale when policies change); expensive | Not used |
| Zero-shot | Simple, no retrieval needed | No brand voice, no fact grounding → higher hallucination | Ablation baseline |
| RAG | Facts injected at inference (fresh); style learned from examples; improves with data | Retrieval quality matters; prompt length grows | **Primary** |
| Nearest-neighbor | Fast, no LLM | Returns old reply verbatim; can't adapt to new scenarios | Ablation baseline |

### Ablation modes

- `rag` — full system (retrieval + KB facts)
- `zero_shot` — LLM only, no retrieval, no KB
- `no_kb` — retrieval examples but KB facts withheld
- `nn` — return the nearest retrieved reply verbatim (no LLM)

---

## Accuracy System — The Core

### What "accurate" means for a suggested reply

Exact match fails because many different replies are equally good ("Your refund of
$42 was issued" and "We've refunded the $42" are both correct with ~zero token overlap).

But a good reply must do more than *resemble* the reference. It must:
1. **Answer every question** the customer asked (coverage)
2. **Use the right facts** — order number, refund window, policy rules (faithfulness)
3. **Not invent facts** — a confident wrong answer is worse than no answer (contradiction detection)
4. **Take the right action** — approve a valid return, decline an expired warranty (action correctness)
5. **Match the customer's tone** — empathize with angry customers, be concise with terse ones (tone)

Surface similarity is blind to all five. BLEU/ROUGE/BERTScore correlate poorly with human
judgment on open-ended generation (established in BLEURT, G-Eval, and BERTScore papers).

### The metric suite

#### Layer 1: Programmatic fact checks (deterministic, no LLM)

The strongest signal. Checks that can't be fooled by an LLM:
- **Entity survival**: order number, item name, price appear correctly
- **Policy numbers**: refund window, warranty duration, shipping fees match KB
- **Decision correctness**: warranty-void scenario doesn't promise replacement; shipped-order
  cancellation isn't falsely confirmed; price-adjustment outside window isn't granted
- **Safety gates**: no forbidden phrases, no prompt leak, signature present, length bounds

#### Layer 2: LLM judge (CoT, rubric-anchored, cross-family)

Fills the semantic gaps that regex can't reach:
- **Coverage** (0–1): for each question the customer asked, did the reply answer it?
- **Faithfulness** (0–1): RAGAS-style claim extraction → support checking against facts
- **Tone** (1–5): register appropriateness judged against the customer's tone
- **Action correctness** (0–1): did the reply commit to the right action?

Judge bias mitigations (per Zheng et al., NeurIPS 2023):
- **Self-preference**: generator and judge use different model families when possible
- **Verbosity bias**: rubric explicitly says "length is not quality"
- **Drift**: temperature=0, JSON-schema output, evidence citation required
- **Auditability**: every judgment includes the rationale and claim-level verdicts

#### Layer 3: Composite scoring with hard gates

```
Score = 0.30·coverage + 0.25·faithfulness + 0.20·action + 0.15·tone + 0.10·entity_accuracy

Gates (penalty ×0.5 each if failed):
  • faithfulness < 0.75
  • contradiction_rate > 0.10  
  • any major programmatic fact-check failure
```

**Why these weights?**
- Coverage (0.30): the #1 job is answering what was asked; an unanswered question forces human intervention
- Faithfulness (0.25): hallucinated facts erode trust and create liability; also gated
- Action (0.20): committing to the wrong action (wrong refund, false cancellation) is a serious error
- Tone (0.15): wrong tone is a real but recoverable error; a human can edit tone in seconds
- Entity accuracy (0.10): verifiable entity checks provide the trustworthy anchor

Weights are validated against human judgments (see next section).

#### Overall system score

- Macro-mean composite over the test set
- Bootstrap 95% confidence interval (2000 resamples)
- Per-intent and per-register breakdown
- Gate failure rates
- Score histogram

---

## Metric Validation

*This is what we care about most* — proving the metric reflects real quality.

### V1. Perturbation / adversarial sensitivity

Take known-good reference replies and corrupt them in specific ways:

| ID | Perturbation | Expected signal |
|----|-------------|----------------|
| P1 | Wrong refund window (5–7 → 3–4 days) | faithfulness ↓ |
| P2 | Wrong order number (swap a digit) | entity_accuracy ↓ |
| P3 | Tone inversion (cold bureaucratic opener) | tone ↓ |
| P4 | Coverage hole (delete one answer) | coverage ↓ |
| P5 | Forbidden phrase ("As an AI...") | gates ↓ |
| P6 | Wrong decision (promise replacement when warranty void) | action ↓ |

**Pass criterion**: corrupted replies score >0.10 below originals, and the targeted
dimension drops the most. If a metric can't detect a known-bad reply, it's decorative.

### V2. Human-label correlation

50 (reply, quality 1–5) pairs hand-labeled under a documented rubric (blind to metric
output). Report Spearman ρ between composite and human labels. Target: ρ ≥ 0.7.

### V3. Inter-judge agreement

Same replies judged by two different models. Report Spearman ρ between judges.
High agreement + high human correlation = the judge is reliable.

---

## Results

See `reports/` for full per-response JSON and markdown reports after running the pipeline.

---

## Trade-offs & Honest Limitations

1. **LLM-judge circularity**: our generator and judge are both LLMs. We address this with:
   - Programmatic fact checks (the strongest signal, zero LLM involvement)
   - Cross-family judging when possible (Gemma vs Gemini)
   - Human calibration of the composite
   - Perturbation tests proving the metric catches known-bad replies

2. **Synthetic data**: cleaner and more regular than real email; stated as a limitation.
   But it enables what real data can't: machine-checkable ground truth per scenario.

3. **Free-tier API rate limits**: dataset build is slow (~90 min). Pre-built dataset ships
   in `data/corpus/`. Mock mode (`LLM_BACKEND=mock`) runs the whole pipeline offline.

4. **Embedding quality**: Gemini embeddings are good but not SOTA; hybrid retrieval
   (semantic + lexical) compensates for embedding blind spots on exact tokens.

5. **Weight subjectivity**: the 0.30/0.25/0.20/0.15/0.10 split encodes our belief about
   what matters in support. We validate it against human judgments and report alternative
   weight correlations.

---

## How AI Tools Were Used

Transparency, as requested:

| Tool | What it did | Human oversight |
|------|------------|----------------|
| **GLM 5.3 (DeepSeek Harness)** | Wrote all code, designed the architecture, authored scenario templates, ran experiments | Human reviewed every design decision, validated test results, authored the weight rationale and metric defense |
| **Gemini Flash Lite** | Realized synthetic emails/replies from structured templates | Programmatic gates validated every output; rejected+regenerated failures |
| **Gemini Flash** | Generated suggested replies (the system under evaluation) | Output evaluated by the metric suite |
| **Gemini/Gemma models** | LLM judge for coverage, faithfulness, tone, action scoring | Cross-family judging; deterministic fact checks as independent signal; human calibration |

All prompts, templates, and seeds are published in this repo for full reproducibility.

---

## Repo Layout

```
data/
  knowledge_base.yaml         # the company's closed world
  corpus/
    train.jsonl               # ~340 (email, reply, facts) pairs for RAG
    test.jsonl                # ~70 held-out test pairs
    demo.jsonl                # 5 examples for quick demo
  dataset_manifest.json       # build report: seed, counts, rejection log
  calibration/
    hand_labels.jsonl          # 50 human labels for metric validation
    rubric.md                  # the rubric used for hand labeling
src/email_reply/
  __init__.py
  __main__.py                 # CLI entry point
  llm.py                      # unified LLM client (Gemini/OpenAI/Mock)
  run.py                      # end-to-end pipeline orchestrator
  dataset/
    synth.py                  # scenario skeleton builder
    build.py                  # LLM realization + gate validation
  gen/
    retrieve.py               # hybrid retriever (semantic + lexical)
    generate.py               # RAG reply generator
  eval/
    facts.py                  # programmatic fact checks (deterministic)
    judge.py                  # LLM judge (coverage, faithfulness, tone, action)
    composite.py              # composite scoring with gates
    validate.py               # metric validation (perturbation, correlation)
reports/
  report_rag.json             # per-response + overall scores
  report_rag.md               # human-readable markdown report
tests/
  test_core.py                # 23 unit tests (zero API calls)
DESIGN.md                     # full technical design document
Makefile                      # build/run commands
pyproject.toml                # project config
```

---

## License

MIT
