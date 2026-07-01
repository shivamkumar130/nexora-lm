import json
import time
from pathlib import Path

from inference.engine.inference_engine import InferenceEngine

from tests.evaluation.evaluation_metrics import (
    exact_match,
    token_overlap,
    precision,
    recall,
    f1_score,
)

ROOT = Path(__file__).resolve().parents[2]

DATASET = ROOT / "tests" / "evaluation" / "coding_dataset.json"

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

OUTPUT = LOG_DIR / "coding_results.json"

engine = InferenceEngine()

dataset = json.loads(
    DATASET.read_text(
        encoding="utf-8"
    )
)

results = []

times = []

exact_total = 0
overlap_total = 0
precision_total = 0
recall_total = 0
f1_total = 0

print("=" * 70)
print("NEXORALM CODING BENCHMARK")
print("=" * 70)

for sample in dataset:

    question = sample["question"]
    expected = sample["expected"]

    start = time.perf_counter()

    prediction = engine.generate(
        question,
        max_new_tokens=60,
        temperature=0.7,
        top_k=20,
    )

    elapsed = time.perf_counter() - start
    times.append(elapsed)

    em = exact_match(prediction, expected)
    overlap = token_overlap(prediction, expected)
    p = precision(prediction, expected)
    r = recall(prediction, expected)
    f1 = f1_score(prediction, expected)

    exact_total += int(em)
    overlap_total += overlap
    precision_total += p
    recall_total += r
    f1_total += f1

    print("\n" + "=" * 70)
    print("Question:")
    print(question)

    print("\nExpected:")
    print(expected)

    print("\nPrediction:")
    print(prediction)

    print("\nMetrics")
    print("Exact Match :", em)
    print("Overlap     :", round(overlap, 4))
    print("Precision   :", round(p, 4))
    print("Recall      :", round(r, 4))
    print("F1 Score    :", round(f1, 4))
    print(f"Time         : {elapsed:.3f} sec")

    results.append(
        {
            "id": sample["id"],
            "category": sample["language"],
            "question": question,
            "expected": expected,
            "prediction": prediction,
            "exact_match": em,
            "overlap": overlap,
            "precision": p,
            "recall": r,
            "f1": f1,
            "time": elapsed,
        }
    )

OUTPUT.write_text(
    json.dumps(
        results,
        indent=4,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)

n = len(results)

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"Questions           : {n}")
print(f"Average Time        : {sum(times)/n:.3f} sec")
print(f"Exact Match         : {(exact_total/n)*100:.2f}%")
print(f"Average Overlap     : {overlap_total/n:.4f}")
print(f"Average Precision   : {precision_total/n:.4f}")
print(f"Average Recall      : {recall_total/n:.4f}")
print(f"Average F1          : {f1_total/n:.4f}")

print(f"\nSaved : {OUTPUT}")
print("=" * 70)