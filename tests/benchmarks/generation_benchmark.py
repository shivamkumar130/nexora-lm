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

# ==========================================================
# PATHS
# ==========================================================

ROOT = Path(__file__).resolve().parents[2]

GENERATION_DATASET = (
    ROOT
    / "tests"
    / "evaluation"
    / "generation_dataset.json"
)

EXPECTED_ANSWERS = (
    ROOT
    / "tests"
    / "evaluation"
    / "expected_answers.json"
)

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = LOG_DIR / "generation_results.json"

# ==========================================================
# LOAD DATA
# ==========================================================

with open(GENERATION_DATASET, "r", encoding="utf-8") as f:
    questions = json.load(f)

with open(EXPECTED_ANSWERS, "r", encoding="utf-8") as f:
    answers = json.load(f)

expected_map = {
    item["id"]: item["answer"]
    for item in answers
}

# ==========================================================
# MODEL
# ==========================================================

engine = InferenceEngine()

# ==========================================================
# BENCHMARK
# ==========================================================

results = []

times = []
word_counts = []

em_scores = []
overlap_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

print("=" * 70)
print("NEXORALM GENERATION BENCHMARK")
print("=" * 70)

for sample in questions:

    question = sample["question"]

    expected = expected_map.get(
        sample["id"],
        ""
    )

    print()
    print("=" * 70)
    print(f"Question : {question}")

    start = time.perf_counter()

    prediction = engine.generate(
        question,
        max_new_tokens=60,
        temperature=1.0,
        top_k=20,
    )

    elapsed = time.perf_counter() - start

    words = len(prediction.split())

    times.append(elapsed)
    word_counts.append(words)

    em = exact_match(
        prediction,
        expected
    )

    overlap = token_overlap(
        prediction,
        expected
    )

    prec = precision(
        prediction,
        expected
    )

    rec = recall(
        prediction,
        expected
    )

    f1 = f1_score(
        prediction,
        expected
    )

    em_scores.append(int(em))
    overlap_scores.append(overlap)
    precision_scores.append(prec)
    recall_scores.append(rec)
    f1_scores.append(f1)

    print("\nExpected:")
    print(expected)

    print("\nPrediction:")
    print(prediction)

    print("\nMetrics")

    print(f"Exact Match : {em}")
    print(f"Overlap     : {overlap:.4f}")
    print(f"Precision   : {prec:.4f}")
    print(f"Recall      : {rec:.4f}")
    print(f"F1 Score    : {f1:.4f}")

    print(f"Time         : {elapsed:.3f} sec")
    print(f"Words        : {words}")

    results.append(
        {
            "id": sample["id"],
            "category": sample["category"],
            "question": question,
            "expected_answer": expected,
            "generated_answer": prediction,
            "generation_time": round(elapsed, 4),
            "word_count": words,
            "metrics": {
                "exact_match": em,
                "token_overlap": round(overlap, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1_score": round(f1, 4),
            },
        }
    )

# ==========================================================
# SAVE
# ==========================================================

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        results,
        f,
        indent=4,
        ensure_ascii=False,
    )

# ==========================================================
# SUMMARY
# ==========================================================

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"Questions              : {len(results)}")

print(f"Average Time (sec)     : {sum(times)/len(times):.3f}")

print(f"Average Words          : {sum(word_counts)/len(word_counts):.2f}")

print(f"Longest Response       : {max(word_counts)}")

print(f"Shortest Response      : {min(word_counts)}")

print()

print(f"Exact Match Accuracy   : {(sum(em_scores)/len(em_scores))*100:.2f}%")

print(f"Average Overlap        : {sum(overlap_scores)/len(overlap_scores):.4f}")

print(f"Average Precision      : {sum(precision_scores)/len(precision_scores):.4f}")

print(f"Average Recall         : {sum(recall_scores)/len(recall_scores):.4f}")

print(f"Average F1 Score       : {sum(f1_scores)/len(f1_scores):.4f}")

print()

print(f"Saved : {OUTPUT_FILE}")

print("=" * 70)