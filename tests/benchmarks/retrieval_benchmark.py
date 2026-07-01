import json
from pathlib import Path

from knowledge.search.hybrid_search import HybridSearch
from tests.evaluation.evaluation_metrics import top_k_hit


ROOT = Path(__file__).resolve().parents[2]

DATASET = (
    ROOT
    / "tests"
    / "evaluation"
    / "retrieval_dataset.json"
)

OUTPUT = (
    ROOT
    / "logs"
    / "retrieval_results.json"
)

search = HybridSearch()

dataset = json.loads(
    DATASET.read_text(
        encoding="utf-8"
    )
)

results = []

top1 = 0
top3 = 0
top5 = 0

print("=" * 60)
print("NEXORALM RETRIEVAL BENCHMARK")
print("=" * 60)

for sample in dataset:

    question = sample["question"]

    expected = sample["expected_chunks"]

    retrieved = search.search(
        question,
        top_k=5
    )

    hit1 = top_k_hit(
        expected,
        retrieved,
        1
    )

    hit3 = top_k_hit(
        expected,
        retrieved,
        3
    )

    hit5 = top_k_hit(
        expected,
        retrieved,
        5
    )

    top1 += int(hit1)
    top3 += int(hit3)
    top5 += int(hit5)

    results.append({

        "question": question,

        "expected_chunks": expected,

        "retrieved_chunks": [
            x["chunk_id"]
            for x in retrieved
        ],

        "top1": hit1,

        "top3": hit3,

        "top5": hit5

    })

    print()

    print(question)

    print("Expected:")

    print(expected)

    print()

    print("Retrieved:")

    for item in retrieved:

        print(
            f"  {item['chunk_id']} "
            f"({item['score']:.4f})"
        )

    print()

    print(
        f"Top1={hit1} "
        f"Top3={hit3} "
        f"Top5={hit5}"
    )

OUTPUT.parent.mkdir(
    exist_ok=True
)

OUTPUT.write_text(

    json.dumps(
        results,
        indent=4
    ),

    encoding="utf-8"

)

n = len(dataset)

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print(
    f"Questions       : {n}"
)

print(
    f"Top-1 Accuracy  : {top1/n:.2%}"
)

print(
    f"Top-3 Accuracy  : {top3/n:.2%}"
)

print(
    f"Top-5 Accuracy  : {top5/n:.2%}"
)

print()

print("Saved:", OUTPUT)