import json
import time
from pathlib import Path

from inference.engine.rag_engine import RAGEngine

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

DATASET = (
    ROOT
    / "tests"
    / "evaluation"
    / "rag_dataset.json"
)

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

OUTPUT = LOG_DIR / "rag_results.json"

# ==========================================================
# LOAD DATASET
# ==========================================================

with open(
    DATASET,
    "r",
    encoding="utf-8"
) as f:

    dataset = json.load(f)

# ==========================================================
# LOAD RAG
# ==========================================================

rag = RAGEngine()

results = []

times = []

exact_scores = []
overlap_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

citation_hits = 0

print("=" * 70)
print("NEXORALM RAG BENCHMARK")
print("=" * 70)

# ==========================================================
# RUN
# ==========================================================

for sample in dataset:

    question = sample["question"]

    expected = sample["expected_answer"]

    print("\n" + "=" * 70)

    print("Question :")
    print(question)

    start = time.perf_counter()

    output = rag.answer(question)

    elapsed = time.perf_counter() - start

    answer = output["answer"]

    citations = output.get(
        "citations",
        []
    )

    # -------------------------------------
    # Evaluation
    # -------------------------------------

    em = exact_match(
        answer,
        expected
    )

    overlap = token_overlap(
        answer,
        expected
    )

    prec = precision(
        answer,
        expected
    )

    rec = recall(
        answer,
        expected
    )

    f1 = f1_score(
        answer,
        expected
    )

    if len(citations) > 0:
        citation_hits += 1

    exact_scores.append(int(em))
    overlap_scores.append(overlap)
    precision_scores.append(prec)
    recall_scores.append(rec)
    f1_scores.append(f1)

    times.append(elapsed)

    # -------------------------------------

    print("\nExpected\n")
    print(expected)

    print("\nGenerated\n")
    print(answer)

    def format_source(src):
        if isinstance(src, dict):
            title = src.get("title", "")
            url = src.get("url", "")
            score = src.get("score", "")
            parts = []
            if title:
                parts.append(title)
            if url:
                parts.append(url)
            if score != "":
                parts.append(f"Score={score}")
            return " | ".join(parts) if parts else str(src)
        return str(src)

    print("\nMetrics")

    print(f"Exact Match : {em}")
    print(f"Overlap     : {overlap:.4f}")
    print(f"Precision   : {prec:.4f}")
    print(f"Recall      : {rec:.4f}")
    print(f"F1 Score    : {f1:.4f}")

    print(f"Time         : {elapsed:.3f} sec")

    # -------------------------------------

    results.append(

        {

            "id": sample["id"],

            "question": question,

            "expected_answer": expected,

            "generated_answer": answer,

            "citations": citations,

            "metrics": {

                "exact_match": em,

                "token_overlap": round(overlap, 4),

                "precision": round(prec, 4),

                "recall": round(rec, 4),

                "f1_score": round(f1, 4)

            },

            "generation_time": round(elapsed, 4)

        }

    )

# ==========================================================
# SAVE
# ==========================================================

with open(
    OUTPUT,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        indent=4,
        ensure_ascii=False
    )

# ==========================================================
# SUMMARY
# ==========================================================

print()

print("=" * 70)
print("SUMMARY")
print("=" * 70)

n = len(results)

print(f"Questions            : {n}")

print(f"Average Time         : {sum(times)/n:.3f} sec")

print()

print(
    f"Exact Match          : {100*sum(exact_scores)/n:.2f}%"
)

print(
    f"Average Overlap      : {sum(overlap_scores)/n:.4f}"
)

print(
    f"Average Precision    : {sum(precision_scores)/n:.4f}"
)

print(
    f"Average Recall       : {sum(recall_scores)/n:.4f}"
)

print(
    f"Average F1           : {sum(f1_scores)/n:.4f}"
)

print()

print(
    f"Citation Coverage    : {100*citation_hits/n:.2f}%"
)

print()

print(f"Saved : {OUTPUT}")

print("=" * 70)