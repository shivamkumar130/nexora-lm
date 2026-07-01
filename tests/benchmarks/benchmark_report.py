import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

LOGS = ROOT / "logs"

DOCS = ROOT / "docs"
DOCS.mkdir(exist_ok=True)

OUTPUT = DOCS / "Phase4_Evaluation_Report.md"


def load_json(filename):
    path = LOGS / filename

    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def average(items, key):

    if not items:
        return 0.0

    values = []

    for item in items:

        if key in item:

            values.append(item[key])

    if not values:
        return 0.0

    return sum(values) / len(values)


# --------------------------------------------------------
# Load Results
# --------------------------------------------------------

generation = load_json("generation_results.json")
retrieval = load_json("retrieval_results.json")
rag = load_json("rag_results.json")
reasoning = load_json("reasoning_results.json")
coding = load_json("coding_results.json")
math = load_json("math_results.json")
hallucination = load_json("hallucination_results.json")

# --------------------------------------------------------
# Generation
# --------------------------------------------------------

generation_f1 = average(generation, "f1")
generation_precision = average(generation, "precision")
generation_recall = average(generation, "recall")

# --------------------------------------------------------
# Retrieval
# --------------------------------------------------------

retrieval_top1 = 0
retrieval_top3 = 0
retrieval_top5 = 0

if retrieval:

    retrieval_top1 = (
        sum(item["top1"] for item in retrieval) /
        len(retrieval)
    )

    retrieval_top3 = (
        sum(item["top3"] for item in retrieval) /
        len(retrieval)
    )

    retrieval_top5 = (
        sum(item["top5"] for item in retrieval) /
        len(retrieval)
    )

# --------------------------------------------------------
# RAG
# --------------------------------------------------------

rag_f1 = average(rag, "f1")

citation_coverage = 0.0

if rag:

    count = 0

    for item in rag:

        if "sources" in item:
            if len(item["sources"]) > 0:
                count += 1

        elif "citations" in item:
            if item["citations"]:
                count += 1

    citation_coverage = count / len(rag)

# --------------------------------------------------------
# Reasoning
# --------------------------------------------------------

reasoning_f1 = average(reasoning, "f1")

# --------------------------------------------------------
# Coding
# --------------------------------------------------------

coding_f1 = average(coding, "f1")

# --------------------------------------------------------
# Mathematics
# --------------------------------------------------------

math_f1 = average(math, "f1")

# --------------------------------------------------------
# Hallucination
# --------------------------------------------------------

hallucination_rate = 0

if hallucination:

    hallucination_rate = (
        sum(
            item["hallucination"]
            for item in hallucination
        ) / len(hallucination)
    )

# --------------------------------------------------------
# Overall Score
# --------------------------------------------------------

overall = (
    generation_f1 +
    rag_f1 +
    reasoning_f1 +
    coding_f1 +
    math_f1 +
    retrieval_top1 +
    citation_coverage +
    (1 - hallucination_rate)
) / 8

# --------------------------------------------------------
# Markdown Report
# --------------------------------------------------------

report = f"""# NexoraLM-0.2 Evaluation Report

---

## Generation

| Metric | Score |
|--------|------:|
| Precision | {generation_precision:.4f} |
| Recall | {generation_recall:.4f} |
| F1 | {generation_f1:.4f} |

---

## Retrieval

| Metric | Score |
|--------|------:|
| Top-1 Accuracy | {retrieval_top1*100:.2f}% |
| Top-3 Accuracy | {retrieval_top3*100:.2f}% |
| Top-5 Accuracy | {retrieval_top5*100:.2f}% |

---

## RAG

| Metric | Score |
|--------|------:|
| Average F1 | {rag_f1:.4f} |
| Citation Coverage | {citation_coverage*100:.2f}% |

---

## Reasoning

Average F1

**{reasoning_f1:.4f}**

---

## Coding

Average F1

**{coding_f1:.4f}**

---

## Mathematics

Average F1

**{math_f1:.4f}**

---

## Hallucination

Hallucination Rate

**{hallucination_rate*100:.2f}%**

---

# Overall Score

**{overall*10:.2f} / 10**

---

Generated Automatically by NexoraLM Phase 4 Evaluation Framework.
"""

OUTPUT.write_text(
    report,
    encoding="utf-8"
)

print("=" * 60)
print("BENCHMARK REPORT GENERATED")
print("=" * 60)
print(f"Saved : {OUTPUT}")
print("=" * 60)