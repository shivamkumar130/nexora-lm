import json
from pathlib import Path

# ==========================================================
# PATHS
# ==========================================================

ROOT = Path(__file__).resolve().parent.parent

TEXT_DIR = ROOT / "processed" / "clean_text"
META_DIR = ROOT / "metadata"

META_DIR.mkdir(
    parents=True,
    exist_ok=True
)

if not TEXT_DIR.exists():
    raise FileNotFoundError(
        f"Directory not found:\n{TEXT_DIR}"
    )

files = sorted(TEXT_DIR.rglob("*.txt"))

print(f"\nFound {len(files)} document(s)\n")

processed = 0

# ==========================================================
# GENERATE METADATA
# ==========================================================

for file in files:

    try:

        text = file.read_text(
            encoding="utf-8",
            errors="ignore"
        )

        stat = file.stat()

        metadata = {

            "document_name": file.stem,

            "file_name": file.name,

            "relative_path": str(
                file.relative_to(TEXT_DIR)
            ),

            "folder_name": file.parent.name,

            "word_count": len(text.split()),

            "character_count": len(text),

            "line_count": len(text.splitlines()),

            "file_size_bytes": stat.st_size,

            "created": stat.st_ctime,

            "modified": stat.st_mtime,

            "embedding_model": "BAAI/bge-small-en-v1.5"

        }

        output = (
            META_DIR
            / file.relative_to(TEXT_DIR)
        ).with_suffix(".json")

        output.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(
            output,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                metadata,
                f,
                indent=4,
                ensure_ascii=False
            )

        processed += 1

        print(f"✓ Metadata Created: {file.name}")

    except Exception as e:

        print(f"✗ Failed: {file.name}")
        print(e)

# ==========================================================
# SUMMARY
# ==========================================================

print("\n" + "=" * 60)
print("METADATA GENERATION COMPLETE")
print("=" * 60)
print(f"Documents Processed : {processed}")
print(f"Documents Found     : {len(files)}")
print(f"Metadata Output     : {META_DIR}")
print("=" * 60)