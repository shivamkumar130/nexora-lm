from pathlib import Path
import shutil

import numpy as np
from sentence_transformers import SentenceTransformer

# ==========================================================
# PATHS
# ==========================================================

ROOT = Path(__file__).resolve().parent.parent

CHUNK_DIR = ROOT / "processed" / "chunks"
EMBED_DIR = ROOT / "processed" / "embeddings"

# ==========================================================
# CHECK INPUT DIRECTORY
# ==========================================================

if not CHUNK_DIR.exists():
    raise FileNotFoundError(
        f"Chunk directory not found:\n{CHUNK_DIR}"
    )

# ==========================================================
# CLEAN OLD EMBEDDINGS
# ==========================================================

if EMBED_DIR.exists():
    shutil.rmtree(EMBED_DIR)

EMBED_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ==========================================================
# LOAD EMBEDDING MODEL
# ==========================================================

print("Loading Embedding Model...")

model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)

# ==========================================================
# FIND CHUNKS
# ==========================================================

chunks = sorted(CHUNK_DIR.rglob("*.txt"))

print(f"Found {len(chunks)} chunk(s)\n")

processed = 0

# ==========================================================
# GENERATE EMBEDDINGS
# ==========================================================

for chunk in chunks:

    try:

        text = chunk.read_text(
            encoding="utf-8",
            errors="ignore"
        ).strip()

        if not text:
            print(f"Skipped Empty Chunk: {chunk.name}")
            continue

        embedding = model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False
        ).astype(np.float32)

        relative = chunk.relative_to(CHUNK_DIR)

        safe_name = "_".join(
            relative.with_suffix("").parts
        )

        output = EMBED_DIR / f"{safe_name}.npy"

        np.save(output, embedding)

        processed += 1

        print(
            f"{chunk.name} -> {output.name}"
        )

    except Exception as e:

        print(f"✗ Failed: {chunk.name}")
        print(e)

# ==========================================================
# SUMMARY
# ==========================================================

print("\n" + "=" * 60)
print("EMBEDDING GENERATION COMPLETE")
print("=" * 60)
print(f"Chunks Found      : {len(chunks)}")
print(f"Chunks Embedded   : {processed}")
print(f"Embedding Model   : BAAI/bge-small-en-v1.5")
print(f"Output Directory  : {EMBED_DIR}")
print("=" * 60)