from pathlib import Path

# ======================================================
# PATHS
# ======================================================

ROOT = Path(__file__).resolve().parent.parent

TEXT_DIR = ROOT / "processed" / "clean_text"

OUTPUT = (
    ROOT
    / "processed"
    / "training_corpus.txt"
)

# ======================================================
# CHECK INPUT
# ======================================================

if not TEXT_DIR.exists():
    raise FileNotFoundError(
        f"Directory not found:\n{TEXT_DIR}"
    )

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

# ======================================================
# FIND DOCUMENTS
# ======================================================

files = sorted(
    TEXT_DIR.rglob("*.txt"),
    key=lambda p: str(p.relative_to(TEXT_DIR))
)

print(f"Found {len(files)} document(s)\n")

all_documents = []
processed = 0
total_words = 0

# ======================================================
# BUILD CORPUS
# ======================================================

for file in files:

    try:

        text = file.read_text(
            encoding="utf-8",
            errors="ignore"
        ).strip()

        if not text:
            print(f"Skipped Empty: {file.name}")
            continue

        total_words += len(text.split())

        document = (
            "\n<DOCUMENT_START>\n"
            f"Document: {file.relative_to(TEXT_DIR)}\n\n"
            f"{text}\n"
            "<DOCUMENT_END>\n"
        )

        all_documents.append(document)
        processed += 1

    except Exception as e:

        print(f"✗ Failed: {file.name}")
        print(e)

# ======================================================
# SAVE CORPUS
# ======================================================

OUTPUT.write_text(
    "\n".join(all_documents),
    encoding="utf-8"
)

# ======================================================
# SUMMARY
# ======================================================

print("\n" + "=" * 60)
print("TRAINING CORPUS CREATED")
print("=" * 60)
print(f"Documents Found      : {len(files)}")
print(f"Documents Processed  : {processed}")
print(f"Total Words          : {total_words:,}")
print(f"Characters           : {OUTPUT.stat().st_size:,}")
print(f"File Size (MB)       : {OUTPUT.stat().st_size / (1024 * 1024):.2f}")
print(f"Output File          : {OUTPUT}")
print("=" * 60)