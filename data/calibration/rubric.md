# Human Labeling Rubric — Reply Quality (1–5)

Used to produce `hand_labels.jsonl` for metric validation (V2).

## Procedure

1. Reviewer reads the **incoming email** and the **ground-truth facts** dict.
2. Reviewer reads the **generated reply** (blind to metric scores).
3. Reviewer assigns a single integer score 1–5 using the anchors below.

## Anchors

| Score | Label | Definition |
|-------|-------|------------|
| 5 | Excellent | Answers every question; all facts correct; appropriate tone; would send as-is |
| 4 | Good | Answers all questions; at most one minor fact imprecision (e.g. rounding); good tone; minor edit needed |
| 3 | Acceptable | Answers most questions; no dangerous fact errors; neutral/slightly off tone; needs editing but usable |
| 2 | Poor | Misses a question OR contains a factual error that could mislead; tone may be off; needs substantial rewrite |
| 1 | Unacceptable | Major factual error (wrong decision, wrong order, invented policy), misses core question, or inappropriate tone; must be rewritten from scratch |

## Specific failure signals

- Wrong order number → automatic ≤ 2
- Promises action contradicting ground truth (e.g. replacement when warranty void) → automatic 1
- Contains "as an AI" / prompt leak → automatic 1
- Missing signature (— The Northwind Team) → cap at 4
- Off-tone (e.g. cold response to furious customer) → cap at 3

## Notes

- Score the reply for this *specific* incoming email and facts, not in the abstract.
- "Would I send this to a customer?" is the ultimate test.
- If in doubt between two scores, prefer the lower one (conservative).
