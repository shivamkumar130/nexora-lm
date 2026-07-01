import re
from pathlib import Path

# ==========================================================
# PATHS
# ==========================================================

ROOT = Path(__file__).resolve().parent.parent

TEXT_DIR = ROOT / "processed" / "clean_text"

if not TEXT_DIR.exists():
    raise FileNotFoundError(
        f"Directory not found:\n{TEXT_DIR}"
    )

files = sorted(TEXT_DIR.rglob("*.txt"))

print(f"\nFound {len(files)} text file(s)\n")

processed = 0

# ==========================================================
# COMPILED REGEX
# ==========================================================

PAGE_MARKER_RE = re.compile(
    r"=+\s*PAGE\s*\d+\s*=+",
    flags=re.IGNORECASE
)

CONTROL_CHAR_RE = re.compile(
    r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]"
)

MULTI_SPACE_RE = re.compile(
    r"[ \t]+"
)

MULTI_BLANK_RE = re.compile(
    r"\n{3,}"
)

# ==========================================================
# CLEAN FILES
# ==========================================================

for file in files:

    try:

        text = file.read_text(
            encoding="utf-8",
            errors="ignore"
        )

        if not text.strip():
            print(f"Skipped Empty: {file.name}")
            continue

        # --------------------------------------------------
        # Normalize line endings
        # --------------------------------------------------

        text = text.replace("\r\n", "\n")
        text = text.replace("\r", "\n")

        cleaned_lines = []
        previous_line = ""

        for line in text.splitlines():

            line = line.strip()

            if not line:
                continue

            # --------------------------------------------------
            # Remove OCR page markers
            # --------------------------------------------------

            line = PAGE_MARKER_RE.sub("", line)

            # --------------------------------------------------
            # Remove control characters
            # (Preserves Unicode symbols such as Δ α β π Ω √ ≤ ≥)
            # --------------------------------------------------

            line = CONTROL_CHAR_RE.sub("", line)

            # --------------------------------------------------
            # Replace tabs with spaces
            # --------------------------------------------------

            line = line.replace("\t", " ")

            # --------------------------------------------------
            # Collapse spaces/tabs only
            # --------------------------------------------------

            line = MULTI_SPACE_RE.sub(" ", line)

            # --------------------------------------------------
            # Fix punctuation spacing
            # --------------------------------------------------

            line = line.replace(" .", ".")
            line = line.replace(" ,", ",")
            line = line.replace(" :", ":")
            line = line.replace(" ;", ";")
            line = line.replace(" !", "!")
            line = line.replace(" ?", "?")
            line = line.replace("( ", "(")
            line = line.replace(" )", ")")

            line = line.strip()

            if len(line) < 2:
                continue

            # --------------------------------------------------
            # Remove consecutive duplicate lines
            # --------------------------------------------------

            if line == previous_line:
                continue

            cleaned_lines.append(line)
            previous_line = line

        cleaned_text = "\n".join(cleaned_lines)

        # --------------------------------------------------
        # Remove excessive blank lines
        # --------------------------------------------------

        cleaned_text = MULTI_BLANK_RE.sub(
            "\n\n",
            cleaned_text
        )

        file.write_text(
            cleaned_text,
            encoding="utf-8"
        )

        processed += 1

        print(f"✓ Cleaned: {file.name}")

    except Exception as e:

        print(f"✗ Failed: {file.name}")
        print(e)

# ==========================================================
# SUMMARY
# ==========================================================

print("\n" + "=" * 60)
print("TEXT CLEANING COMPLETE")
print("=" * 60)
print(f"Files Found      : {len(files)}")
print(f"Files Processed  : {processed}")
print("=" * 60)