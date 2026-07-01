from pathlib import Path
import shutil

# ==========================================================
# PATHS
# ==========================================================

ROOT = Path(__file__).resolve().parent.parent

TEXT_DIR = ROOT / "processed" / "clean_text"
CHUNK_DIR = ROOT / "processed" / "chunks"

# ==========================================================
# SETTINGS
# ==========================================================

CHUNK_SIZE = 320
OVERLAP = 64
MIN_CHUNK_SIZE = 80

# ==========================================================
# CLEAN OLD CHUNKS
# ==========================================================

if CHUNK_DIR.exists():
    shutil.rmtree(CHUNK_DIR)

CHUNK_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ==========================================================
# LOAD FILES
# ==========================================================

if not TEXT_DIR.exists():
    raise FileNotFoundError(
        f"Directory not found:\n{TEXT_DIR}"
    )

files = sorted(TEXT_DIR.rglob("*.txt"))

print(f"\nFound {len(files)} document(s)\n")

processed = 0
total_chunks = 0

# ==========================================================
# PROCESS DOCUMENTS
# ==========================================================

for file in files:

    try:

        text = file.read_text(
            encoding="utf-8",
            errors="ignore"
        )

        text = text.strip()

        if not text:
            print(f"Skipped Empty File: {file.name}")
            continue

        # --------------------------------------------------
        # Preserve paragraph structure
        # --------------------------------------------------

        text = text.replace("\r\n", "\n")
        text = text.replace("\r", "\n")

        lines = []

        for line in text.splitlines():

            line = line.strip()

            if line:
                lines.append(line)

        text = "\n".join(lines)

        words = text.split()

        if len(words) < MIN_CHUNK_SIZE:
            print(f"Skipped Small File: {file.name}")
            continue

        # --------------------------------------------------
        # Preserve relative folder structure
        # --------------------------------------------------

        relative = file.relative_to(TEXT_DIR)

        save_folder = (
            CHUNK_DIR
            / relative.parent
            / file.stem
        )

        save_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        start = 0
        chunk_number = 1

        while start < len(words):

            end = min(
                start + CHUNK_SIZE,
                len(words)
            )

            chunk_words = words[start:end]

            if not chunk_words:
                break

            if len(chunk_words) >= MIN_CHUNK_SIZE:

                chunk_text = (
                    f"Document: {file.stem}\n"
                    f"Chunk: {chunk_number}\n"
                    f"Start Word: {start}\n"
                    f"End Word: {end}\n"
                    f"Words: {len(chunk_words)}\n\n"
                    + " ".join(chunk_words)
                )

                chunk_path = (
                    save_folder
                    / f"chunk_{chunk_number}.txt"
                )

                chunk_path.write_text(
                    chunk_text,
                    encoding="utf-8"
                )

                total_chunks += 1
                chunk_number += 1

            if end >= len(words):
                break

            start += CHUNK_SIZE - OVERLAP

        processed += 1

        print(
            f"{file.relative_to(TEXT_DIR)}"
            f" -> {chunk_number - 1} chunk(s)"
        )

    except Exception as e:

        print(f"✗ Failed: {file.name}")
        print(e)

# ==========================================================
# SUMMARY
# ==========================================================

print("\n" + "=" * 60)
print("CHUNK GENERATION COMPLETE")
print("=" * 60)
print(f"Documents Found      : {len(files)}")
print(f"Documents Processed  : {processed}")
print(f"Total Chunks         : {total_chunks}")
print(f"Chunk Size           : {CHUNK_SIZE}")
print(f"Overlap              : {OVERLAP}")
print(f"Minimum Chunk Size   : {MIN_CHUNK_SIZE}")
print(f"Output Directory     : {CHUNK_DIR}")
print("=" * 60)