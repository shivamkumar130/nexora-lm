import json
import time
from pathlib import Path

from inference.engine.inference_engine import InferenceEngine

ROOT = Path(__file__).resolve().parents[2]

DATASET = ROOT / "tests" / "evaluation" / "hallucination_dataset.json"

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

OUTPUT = LOG_DIR / "hallucination_results.json"

engine = InferenceEngine()

dataset = json.loads(
    DATASET.read_text(
        encoding="utf-8"
    )
)

results = []

hallucinations = 0

correct_unknown = 0

times = []

print("=" * 70)
print("NEXORALM HALLUCINATION BENCHMARK")
print("=" * 70)

for sample in dataset:

    question = sample["question"]

    expected = sample["expected"]

    start = time.perf_counter()

    prediction = engine.generate(
        question,
        max_new_tokens=40,
        temperature=0.7,
        top_k=20
    ).strip()

    elapsed = time.perf_counter() - start

    times.append(elapsed)

    is_unknown = prediction.lower().strip() == expected.lower().strip()

    if is_unknown:
        correct_unknown += 1
    else:
        hallucinations += 1

    print("\n" + "=" * 70)

    print("Question:")
    print(question)

    print("\nExpected:")
    print(expected)

    print("\nPrediction:")
    print(prediction)

    print("\nHallucination :", not is_unknown)

    print(f"Time           : {elapsed:.3f} sec")

    results.append(
        {
            "id": sample["id"],
            "question": question,
            "expected": expected,
            "prediction": prediction,
            "hallucination": not is_unknown,
            "time": elapsed
        }
    )

OUTPUT.write_text(
    json.dumps(
        results,
        indent=4,
        ensure_ascii=False
    ),
    encoding="utf-8"
)

n = len(results)

hallucination_rate = hallucinations / n

unknown_accuracy = correct_unknown / n

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"Questions              : {n}")

print(f"Average Time           : {sum(times)/n:.3f} sec")

print(f"Correct 'I don't know' : {unknown_accuracy*100:.2f}%")

print(f"Hallucination Rate     : {hallucination_rate*100:.2f}%")

print()

print(f"Saved : {OUTPUT}")

print("=" * 70)