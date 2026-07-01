from pathlib import Path
import json

import faiss
import numpy as np

# ==========================================================
# PATHS
# ==========================================================

ROOT = Path(__file__).resolve().parents[2]

EMBED_DIR = ROOT / "datasets" / "processed" / "embeddings"
CHUNK_DIR = ROOT / "datasets" / "processed" / "chunks"

FAISS_DIR = ROOT / "knowledge" / "faiss"
FAISS_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# CHECK DIRECTORIES
# ==========================================================

if not EMBED_DIR.exists():
    raise FileNotFoundError(
        f"Embedding directory not found:\n{EMBED_DIR}"
    )

if not CHUNK_DIR.exists():
    raise FileNotFoundError(
        f"Chunk directory not found:\n{CHUNK_DIR}"
    )

embedding_files = sorted(EMBED_DIR.glob("*.npy"))

if not embedding_files:
    raise RuntimeError(
        "No embedding files found."
    )

print(f"Found {len(embedding_files)} embedding(s)\n")

vectors = []
chunk_map = {}

# ==========================================================
# LOAD EMBEDDINGS
# ==========================================================

for idx, emb_file in enumerate(embedding_files):

    try:

        vector = np.load(emb_file).astype(np.float32)

        vectors.append(vector)

        chunk_id = emb_file.stem

        # --------------------------------------------------
        # Example:
        # Ai_pdf_nexora_Ai_notes1_v1_chunk_12
        # --------------------------------------------------

        try:
            base_name, chunk_number = chunk_id.rsplit(
                "_chunk_",
                1
            )
        except ValueError:
            print(f"Invalid embedding filename: {emb_file.name}")
            continue

        parts = base_name.split("_")

        if len(parts) < 3:
            print(f"Invalid embedding filename: {emb_file.name}")
            continue

        category = parts[0]
        source_type = parts[1]
        document_name = "_".join(parts[2:])

        file_name = f"chunk_{chunk_number}.txt"

        chunk_path = (
            CHUNK_DIR
            / category
            / source_type
            / document_name
            / file_name
        )

        if not chunk_path.exists():
            print(f"WARNING: Missing chunk file -> {chunk_path}")

        chunk_map[str(idx)] = {
            "chunk_id": chunk_id,
            "chunk_path": str(
                chunk_path.relative_to(ROOT)
            )
        }

    except Exception as e:

        print(f"Failed: {emb_file.name}")
        print(e)

# ==========================================================
# BUILD MATRIX
# ==========================================================

if not vectors:
    raise RuntimeError(
        "No valid embeddings were loaded."
    )

matrix = np.vstack(vectors).astype(np.float32)

dimension = matrix.shape[1]

faiss.normalize_L2(matrix)

index = faiss.IndexFlatIP(dimension)

index.add(matrix)

# ==========================================================
# SAVE
# ==========================================================

faiss.write_index(
    index,
    str(FAISS_DIR / "index.faiss")
)

with open(
    FAISS_DIR / "chunk_map.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        chunk_map,
        f,
        indent=4,
        ensure_ascii=False
    )

# ==========================================================
# SUMMARY
# ==========================================================

print("\n" + "=" * 60)
print("FAISS INDEX CREATED SUCCESSFULLY")
print("=" * 60)
print(f"Embeddings Loaded : {len(vectors)}")
print(f"Vector Dimension  : {dimension}")
print(f"Index File        : {FAISS_DIR / 'index.faiss'}")
print(f"Chunk Map         : {FAISS_DIR / 'chunk_map.json'}")
print("=" * 60)