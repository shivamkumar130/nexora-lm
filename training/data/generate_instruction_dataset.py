import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CHUNK_ROOT = ROOT / "datasets" / "processed" / "chunks"

OUTPUT_DIR = ROOT / "datasets" / "instruction"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "instruction_train.json"

print("ROOT :", ROOT)
print("CHUNK ROOT :", CHUNK_ROOT)
print("Exists :", CHUNK_ROOT.exists())

# ----------------------------------------------------------
# Instruction Templates
# ----------------------------------------------------------

SHORT_TEMPLATES = [

    "What is {topic}?",
    "Define {topic}.",
    "Give the definition of {topic}.",

]

MEDIUM_TEMPLATES = [

    "Explain {topic}.",
    "Describe {topic}.",
    "Give a short note on {topic}.",
    "Summarize {topic}.",
    "Write about {topic}.",

]

LONG_TEMPLATES = [

    "Explain {topic} in detail.",
    "Provide a detailed explanation of {topic}.",
    "Write detailed notes on {topic}.",
    "Describe the concept of {topic}.",
    "Explain everything about {topic}.",

]

FEATURE_TEMPLATES = [

    "What are the important points of {topic}?",
    "List the key features of {topic}.",
    "Explain the main ideas of {topic}.",
    "What should I know about {topic}?",

]

APPLICATION_TEMPLATES = [

    "What are the applications of {topic}?",
    "Where is {topic} used?",
    "Why is {topic} important?",

]

ALL = [

    ("short", SHORT_TEMPLATES),

    ("medium", MEDIUM_TEMPLATES),

    ("long", LONG_TEMPLATES),

    ("feature", FEATURE_TEMPLATES),

    ("application", APPLICATION_TEMPLATES)

]

dataset = []

sample_id = 1

# ----------------------------------------------------------
# Read every chunk
# ----------------------------------------------------------

for chunk_file in CHUNK_ROOT.rglob("*.txt"):

    try:

        text = chunk_file.read_text(
            encoding="utf-8",
            errors="ignore"
        ).strip()

    except Exception:

        continue

    if not text:

        continue

    lines = text.splitlines()

    topic = chunk_file.parent.name.replace("_", " ")

    # ------------------------------------------------------
    # Read document name
    # ------------------------------------------------------

    for line in lines:

        if line.lower().startswith("document:"):

            topic = line.split(":", 1)[1].strip()

            break

    # ------------------------------------------------------
    # Remove metadata
    # ------------------------------------------------------

    body = []

    for line in lines:

        lower = line.lower()

        if lower.startswith("document:"):
            continue

        if lower.startswith("chunk:"):
            continue

        if lower.startswith("start word:"):
            continue

        if lower.startswith("end word:"):
            continue

        if lower.startswith("words:"):
            continue

        body.append(line)

    body = "\n".join(body).strip()

    if len(body) < 120:
        continue

    # ------------------------------------------------------
    # Build different answer sizes
    # ------------------------------------------------------

    paragraphs = [

        p.strip()

        for p in body.split("\n\n")

        if p.strip()

    ]

    sentences = []

    for p in paragraphs:

        sentences.extend(

            [

                s.strip()

                for s in p.split(".")

                if len(s.strip()) > 15

            ]

        )

    if len(sentences) == 0:
        continue

    short_answer = ". ".join(sentences[:2]).strip()

    if short_answer:
        short_answer += "."

    medium_answer = ". ".join(sentences[:5]).strip()

    if medium_answer:
        medium_answer += "."

    long_answer = body

    feature_answer = "\n".join(paragraphs[:2])

    application_answer = "\n".join(paragraphs[-2:])

    answers = {

        "short": short_answer,

        "medium": medium_answer,

        "long": long_answer,

        "feature": feature_answer,

        "application": application_answer

    }

    # ------------------------------------------------------
    # Generate instructions
    # ------------------------------------------------------

    for category, templates in ALL:

        for template in templates:

            answer = answers[category].strip()

            if len(answer) < 40:
                continue

            dataset.append({

                "id": sample_id,

                "instruction": template.format(
                    topic=topic
                ),

                "input": "",

                "output": answer

            })

            sample_id += 1

# ----------------------------------------------------------
# Shuffle
# ----------------------------------------------------------

random.shuffle(dataset)

# ----------------------------------------------------------
# Save
# ----------------------------------------------------------

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        dataset,
        f,
        indent=4,
        ensure_ascii=False
    )

print("=" * 60)
print("Instruction Dataset Generated")
print("=" * 60)
print("Samples :", len(dataset))
print("Saved   :", OUTPUT_FILE)
print("=" * 60)