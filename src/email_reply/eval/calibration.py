"""Generate synthetic hand labels for metric calibration.

In a real deployment, these would be created by a human reviewer following
data/calibration/rubric.md. For this project, we generate calibration labels
from the dataset's reference replies (which are known-good, score 4-5) and
from perturbation-corrupted replies (which are known-bad, score 1-3).

This gives us a set of (reply, human_score) pairs with ground-truth quality
variance that the composite metric should correlate with.
"""
import json
import random
from pathlib import Path

import yaml

from .validate import PERTURBATIONS

DATA = Path(__file__).resolve().parents[3] / "data"


def generate_hand_labels(test_path: Path = DATA / "corpus" / "test.jsonl",
                         n: int = 50, seed: int = 7) -> list[dict]:
    """Create calibration labels from reference + perturbed replies."""
    rng = random.Random(seed)
    kb = yaml.safe_load((DATA / "knowledge_base.yaml").read_text())

    with open(test_path) as f:
        pairs = [json.loads(line) for line in f]

    if len(pairs) < 10:
        # fallback: use what we have
        pass

    labels = []

    # Good replies (reference): score 4 or 5
    good_sample = rng.sample(pairs, min(n // 2, len(pairs)))
    for p in good_sample:
        score = rng.choice([4, 4, 5, 5, 5])  # reference replies are generally good
        labels.append({
            "id": p["id"],
            "intent": p["intent"],
            "register": p.get("register", ""),
            "incoming": p["incoming"][:300],
            "reply": p["reply"],
            "human_score": score,
            "source": "reference",
            "note": "Reference reply from dataset (known-good)",
        })

    # Corrupted replies: score 1-3 depending on corruption severity
    severity_map = {
        "P1": (1, 2),   # wrong fact -> bad
        "P2": (2, 3),   # wrong order number -> medium-bad
        "P3": (2, 3),   # tone inversion -> medium
        "P4": (2, 3),   # coverage hole -> medium
        "P5": (1, 1),   # forbidden phrase -> very bad
        "P6": (1, 1),   # wrong decision -> very bad
    }

    corrupt_sample = rng.sample(pairs, min(n - len(labels), len(pairs)))
    for p in corrupt_sample:
        pert = rng.choice(PERTURBATIONS)
        corrupted = pert.fn(p["reply"], p["facts"], kb)
        lo, hi = severity_map.get(pert.id, (1, 3))
        score = rng.randint(lo, hi)
        labels.append({
            "id": p["id"] + f"_corrupt_{pert.id}",
            "intent": p["intent"],
            "register": p.get("register", ""),
            "incoming": p["incoming"][:300],
            "reply": corrupted,
            "human_score": score,
            "source": f"perturbed:{pert.id}:{pert.name}",
            "note": f"Corrupted via {pert.name}; expected signal: {pert.expected_signal}",
        })

    rng.shuffle(labels)
    return labels[:n]


def save_hand_labels(labels: list[dict], path: Path | None = None):
    path = path or DATA / "calibration" / "hand_labels.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for label in labels:
            f.write(json.dumps(label, ensure_ascii=False) + "\n")
    print(f"Saved {len(labels)} hand labels to {path}")


if __name__ == "__main__":
    labels = generate_hand_labels()
    save_hand_labels(labels)
