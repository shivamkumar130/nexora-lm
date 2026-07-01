import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CHUNK_MAP = ROOT / "knowledge" / "faiss" / "chunk_map.json"
OUTPUT_DIR = ROOT / "tests" / "evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with open(CHUNK_MAP, "r", encoding="utf-8") as f:
    chunk_map = json.load(f)

retrieval_dataset = []
evaluation_dataset = []
expected_answers = []

question_id = 1

for value in chunk_map.values():

    chunk_id = value["chunk_id"]
    chunk_path = ROOT / value["chunk_path"]

    if not chunk_path.exists():
        continue

    text = chunk_path.read_text(
        encoding="utf-8",
        errors="ignore"
    ).strip()

    if not text:
        continue

    lines = text.splitlines()

    document_name = ""

    for line in lines:
        if line.lower().startswith("document:"):
            document_name = line.split(":", 1)[1].strip()
            break

    content_start = 0

    for i, line in enumerate(lines):
        if line.strip() == "":
            content_start = i + 1
            break

    body = "\n".join(lines[content_start:]).strip()

    if len(body) < 100:
        continue

    first_sentence = body.split(".")[0].strip()

    if len(first_sentence) < 20:
        first_sentence = body[:120].replace("\n", " ")

    question = f"What is {document_name}?"

    retrieval_dataset.append(
        {
            "id": question_id,
            "question": question,
            "expected_chunks": [
                chunk_id
            ]
        }
    )

    evaluation_dataset.append(
        {
            "id": question_id,
            "question": question,
            "ground_truth": body
        }
    )

    expected_answers.append(
        {
            "id": question_id,
            "question": question,
            "answer": first_sentence
        }
    )

    question_id += 1

print(f"Generated {question_id-1} evaluation samples.")

with open(
    OUTPUT_DIR / "retrieval_dataset.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        retrieval_dataset,
        f,
        indent=4,
        ensure_ascii=False
    )

with open(
    OUTPUT_DIR / "evaluation_dataset.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        evaluation_dataset,
        f,
        indent=4,
        ensure_ascii=False
    )

with open(
    OUTPUT_DIR / "expected_answers.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        expected_answers,
        f,
        indent=4,
        ensure_ascii=False
    )

print("Done.")