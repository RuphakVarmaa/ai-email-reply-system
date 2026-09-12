# Design — AI Email Suggested-Response System

## 0. The one-paragraph version

A customer-support mailbox (fictional "Northwind Supply Co.") has 426 past incoming
emails paired with the replies a human agent actually sent. A new email arrives; we
retrieve the most similar past conversations, and an LLM drafts a suggested reply
grounded in (a) those retrieved examples and (b) a **knowledge base of company facts**
(policies, prices, shipping windows). The reply is then scored by a **multi-metric
evaluation system** that measures the things that actually make a suggested reply good —
fact fidelity to the company's world, coverage of what the customer asked, tone/register
appropriateness — rather than string overlap with a reference. The whole pipeline is
validated end-to-end: the metric suite is itself validated against ~50 human-labeled
judgments, adversarially corrupted replies are detected as bad, and ablations show RAG
grounding measurably improves scores vs. zero-shot.

## 1. Why not just compare to the "correct" reply with exact match / ROUGE?

The task says it: exact match is too strict. But it's worth being precise about *why*
surface-overlap fails for replies, because that determines the metric design:

1. **The reference reply is one of many acceptable replies.** "Your refund of $42.00
   was issued today, expect it in 5–7 business days" and "We've refunded the $42 — it
   should land on your card within a week" are both perfect answers with ~zero token
   overlap. A reply-metric must be sensitive to *meaning*, not tokens (G-Eval, BERTScore
   literature: BLEU/ROUGE correlate poorly with humans on open-ended generation).
2. **A good suggested reply must do more than resemble the past.** It must (a) answer
   every question asked, (b) use the *right facts* (order #, refund policy window,
   shipping time), (c) match the sender's tone, and (d) not invent facts. String
   similarity is blind to all four. In our domain, a reply that confidently gives a
   wrong refund window is a *failure* even if it's 95% similar to the reference.
3. **LLM-as-judge alone is unreliable.** Judge biases are well documented: position
   bias, verbosity bias, self-preference (a Gemini judge favors Gemini outputs). We
   mitigate each explicitly (§5).

So the accuracy system is a **suite**: reference-free fact-level metrics computed from
programmatic extraction + a cross-family LLM judge + reference-based semantic
similarity, combined into a composite score with hard gates. Every per-response score
ships with the evidence behind it (which facts were checked, which claims were
unsupported).

## 2. What "accurate" means for a suggested reply — formalized

For each test email, our dataset builder stores a machine-checkable **facts dict**:
the order number, item, price, refund window in days, shipping window, policy rules
that apply, the customer's name, etc. This is possible because we *author* the scenario
templates — we know ground truth (see §3). The evaluator then scores a reply against
the *world* (facts dict + knowledge base), not just the reference text.

Let a generated reply r contain a set of **claims** C(r) extracted by an LLM claim
extractor (each claim atomic, e.g. "refund window is 7 days"). Let KB be the knowledge
base. Each claim is checked for **support**: supported, contradicted, or unverifiable.

- **Fact coverage** = |{q ∈ Q answered in r}| / |Q|, where Q = questions/requests in the incoming email (LLM-extracted, human-verified).
- **Faithfulness** = |supported(C(r))| / |C(r)| (RAGAS-style). Unsupported ⇒ hallucination risk.
- **Contradiction rate** = |contradicted(C(r))| / |C(r)|. A reply asserting the wrong refund window is a contradiction, not merely "unsupported".
- **Action correctness** (for action scenarios): did the reply state the correct action for this situation given KB? (LLM-judged against KB with CoT, cross-checked with programmatic checks where possible.)
- **Tone/register score**: judge score of tone appropriateness vs. the sender's tone and the company's brand voice guide (from KB).
- **Conciseness/verbosity gate** (not score-blending — a gate, see §5).
- **Semantic similarity to reference** (embedding cosine): a *secondary* sanity signal, reported but down-weighted, because acceptable replies vary hugely.

Composite:
```
Score(r) = 0.30·coverage + 0.25·faithfulness + 0.20·action_correctness + 0.15·tone + 0.10·similarity
subject to gates: faithfulness ≥ 0.85, contradiction ≤ 0.05, no forbidden phrase, no leaked prompt.
Score_capped = Score · Π gates (gate failure ⇒ multiply by a penalty factor, not zero, so gradients survive)
```

Wait — one gate deserves special care: a reply that answers nothing should not pass just because it's short and truthful. Coverage drives the score with heaviest weight; gates are binary multipliers for safety-critical dimensions (hallucinated facts are dangerous regardless of coverage).

Overall system score = macro-average over test set + bootstrap 95% CI + per-intent breakdown + gate failure rates.

### Why these weights?

Weights are a claim about what matters in a support mailbox, and we validate them (see §6):
- **Coverage (0.30)** — the #1 job of a reply is to answer the questions asked. A reply that skips a question forces a human to intervene, which is exactly what a suggestion system must avoid.
- **Faithfulness (0.25)** — hallucinated facts (wrong order status, invented prices) are the worst failure mode in customer support: they're confidently wrong, they erode trust, and they can create legal exposure. So heavily weighted but also gated.
- **Action correctness (0.(ec)20)** — in our domain a reply often commits to an action ("we've issued a refund", "we've reshipped the item"). Committing to the wrong action, or failing to commit to the one the reference made, is a serious error.
- **Tone (0.15)** — wrong tone (overly blunt to a grieving customer, overly casual to a corporate buyer) is a real but recoverable error; a human can edit tone in 5 seconds, so it's scored but not gated.
- **Similarity (0.10)** — useful regression signal (catches "plausible but irrelevant" replies), but down-weighted since many very different replies are equally right.
Subjective? Yes — so we **calibrate weights against human judgments** (§6): we hand-label 50 (reply, quality) pairs, then fit the composite (linear blend over normalized judge components) and confirm ranking correlation (Spearman ρ) with human labels is high (target ρ ≥ 0.7), and compare against alternative weightings.

## 3. Dataset: honesty and representativeness

**Source: fully synthetic, but with three degrees of grounding.** We own no real email
corpus we could publish (real replies = private data). Options considered:

| Option | Pros | Cons | Verdict |
|---|---|---|---|
| Enron (public, real) | realism; thread structure | replies sparse, 2001-era corporate, no ground-truth facts, licensing fine but **no fact anchors** → can't validate factual accuracy of replies at all; cleaning Enron into clean (email → reply) pairs with dedup is a known mess | used as *style prior* only (rejected as primary) |
| Public support-email corpora (e.g., HuggingFace email-threads) | cheap | tiny, toy, same no-facts problem | rejected |
| **Synthetic, LLM-realized from fact templates** | full ground truth per pair; controllable difficulty; reproducible with a seed; can encode failure modes deliberately | realism risk: LLM-templated emails can read formulaic | **chosen** |

Representativeness defense: a customer-support mailbox is a small closed world —
**intents** (refund, shipping, exchange, bug report, complaint, question, greeting...),
**facts** (order IDs, prices, dates, policies), and **registers** (calm, furious, terse,
chatty). Our generator samples across 10 intent families × 6 registers × varied customer
personas, with `numpy`-seeded randomness so every pair is reproducible from
`dataset_manifest.json`. Each pair records its scenario template id, so an evaluator
can answer "is the metric sensitive to the *right* things?" per intent.

Every pair also stores a `facts` dict (ground-truth facts for that scenario) and the
KB anchor lines the reference reply relies on. This is the honest data flywheel:
- **Generation grounding**: the RAG generator retrieves past pairs *and* KB anchors.
- **Evaluation grounding**: fact metrics check the generated reply against `facts` + KB.
- **Validation**: since ground truth is known, we can corrupt replies (change refund
  window 7→3 days) and confirm the evaluator flags it (a **perturbation test** — if a
  "good" metric doesn't punish known-bad replies, it's measuring nothing).

**Scale**: 426 pairs: ~350 train (RAG corpus) + 76 held-out test. Also a 5-pair demo set.

**Honest limitations section in README**: synthetic emails are cleaner than real mail
(no quoted signatures, no forwarding chains); intent mix is ours to choose; LLM-realized
text may carry LLM phrasing tics; the KB is small but complete for the world. We
publish the generator seeds and templates so anyone can regenerate or extend.

## 4. Response generator (Gen-AI)

```
incoming email
   │
   ├─ normalize (lowercase for matching only; keep subject/body distinct)
   ├─ embed (OpenAI text-embedding-3-small) → cosine top-k past pairs (k=6)
   ├─ plus BM25-ish lexical overlap filter → hybrid retrieval (dedupe)
   │
   ├─ knowledge base facts: look up anchor facts for retrieved scenarios → resolve entities (order # → order record with price, ship window, status)
   │
   └─ prompt = system(brand voice + KB facts for this thread) + few-shot k retrieved (incoming⇒reply) + the new email
          → LLM (default gemini-3.5-flash; OpenAI gpt-5.6-sol alternative) → suggested reply
```

Trade-offs (fuller version in README):
- **RAG over few-shot-only / zero-shot**: grounding in retrieved past pairs gives the model the *company's way* of phrasing replies, and KB anchors give correct facts. RAG also makes the system improve as you add data without retraining.
- **RAG over fine-tuning**: for 350 pairs, fine-tuning teaches style but risks memorizing facts; we want facts injected at inference time (they change! prices/policies change weekly). Cost: a fine-tune could capture more brand voice than a prompt can — a real trade-off we state, and we ablate zero-shot vs. RAG to quantify the gap.
- **Hybrid retrieval (semantic + lexical)**: emails contain order numbers, SKUs — exact tokens embeddings blur. Hybrid catches ID-heavy queries.
- **Model choice**: default Gemini Flash (fast, cheap, strong); judge uses **different family** (GPT) to avoid self-preference bias — see §5.

## 5. Accuracy system architecture

```
generated reply ──┬─→ 1. Fact metrics (programmatic + LLM claim checker)
                  │      coverage, faithfulness (claim-level, RAGAS-style), contradiction rate,
                  │      action correctness, entity accuracy (order # etc. from facts dict)
                  ├─→ 2. Cross-family LLM judge (GPT) with CoT, rubric, position-randomized
                  │      tone/register, action correctness tie-break, overall 1–5
                  ├─→ 3. Reference-based: embedding similarity (secondary), keyword overlap
                  └─→ 4. Gates: no KB-contradicting claims, no forbidden phrases, no prompt leak, length in bounds

Composite per-response score + evidence bundle (per-fact check results, judge rationale)
Overall: weighted mean over test set, bootstrap 95% CI, per-intent table, gate failure rate
```

**Judge-bias mitigations** (from the LLM-as-judge literature):
- **Self-preference**: generator Gemini, judge GPT (different families).
- **Position bias**: not pairwise A/B here, but rubric anchors randomize order of evidence presentation.
- **Verbosity bias**: judge prompted with explicit "length is not quality" anchor; conciseness gate independent.
- **Judge drift**: temperature 0; JSON-schema output; every judgment includes a one-line evidence citation, so a human can audit 10 judgments in a minute.

**Validation of the metric system itself** (this is the core differentiator):
1. **Human-label correlation**: 50 (reply, 1–5) pairs hand-labeled by us (documented procedure: blind to metric scores, rubric-only). Spearman ρ between composite and human labels, plus per-metric correlations. Target ρ ≥ 0.7.
2. **Perturbation/adversarial sensitivity**: take known-good replies, corrupt one fact (wrong refund window), confirm score drops materially (>0.10). A metric that can't detect injected hallucinations is decorative.
3. **Ablation sensitivity**: zero-shot vs. RAG pipeline — if the metric says RAG is better, and retrieval demonstrably feeds the right KB facts, the metric is tracking something real.
4. **Discriminative power**: the same generator with degraded retrieval (k=0) must score lower on faithfulness — confirming faithfulness responds to grounding.
5. **Inter-judge agreement**: two judge models (GPT and Gemini-Pro) score the same 50 replies; report agreement (Pearson/Spearman between judges) — a proxy for judge reliability. If cross-family judges agree with each other AND with humans, the judge isn't just noise.

## 6. Run modes

```
make build-dataset      # regenerate the corpus from templates + seed (offline, deterministic)
make demo               # single email → suggestion → score (one curl-able example)
make run                # full test set: generate + evaluate + reports
make ablate             # zero-shot vs RAG vs no-KB ablations (subset for cost)
make validate-metric    # human-label correlation + perturbation + judges agreement
make test              # pytest unit tests
```

API keys via env (`GEMINI_API_KEY`, `OPENAI_API_KEY`); no keys needed for dataset build or tests; a **mock LLM mode** (`LLM_BACKEND=mock`) lets the whole pipeline run offline for CI.

## 7. Repo layout

```
data/
  corpus/train.jsonl          # (incoming, reply, facts, kb_anchors, intent, register, template_id)
  corpus/test.jsonl
  knowledge_base.yaml         # the company world: policies, catalog, templates
  calibration/hand_labels.jsonl   # 50 hand labels
evals/
  report.json / report.md     # per-response + overall, CI, per-intent, gates
  ablations.json
  metric_validation.json
src/  (package: `email_reply`)
  dataset/build.py, dataset/synth.py
  gen/ retrieve.py, generate.py, llm.py
  eval/ facts.py, judge.py, metrics.py, composite.py, perturb.py, validate.py
  cli.py
scripts/  demo.sh, run_all.sh
tests/   test_*.py
```

## 8. Risks & honest trade-offs

- **LLM-judge circularity**: our generator and judge are LLMs; we address with cross-family judging + programmatic fact checks (the strongest signal, no LLM needed for entity/date checks) + human calibration.
- **Synthetic data realism**: mitigated by register/persona variety + fact-checked realism pass; stated as a limitation.
- **Cost**: full test run ≈ 76 generations + ~76×6 judge/extract calls ≈ manageable; ablations on a 24-pair subset; mock mode for CI.
- **Judge variance**: temperature 0, JSON outputs, single-call-per-judgment; we report CI over test set, not over judge calls.
