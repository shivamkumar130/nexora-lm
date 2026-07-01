from pathlib import Path
import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[2]

FAISS_DIR = ROOT / "knowledge" / "faiss"

INDEX_FILE = FAISS_DIR / "index.faiss"
MAP_FILE = FAISS_DIR / "chunk_map.json"

index = faiss.read_index(str(INDEX_FILE))

with open(MAP_FILE, "r", encoding="utf-8") as f:
    chunk_map = json.load(f)

model = SentenceTransformer("BAAI/bge-small-en-v1.5")


def retrieve(query, top_k=5):

    embedding = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype(np.float32)

    scores, indices = index.search(
        embedding,
        top_k
    )

    results = []

    for idx, score in zip(indices[0], scores[0]):

        if idx == -1:
            continue

        item = chunk_map.get(str(idx))

        if item is None:
            continue

        results.append({
            "chunk_id": item["chunk_id"],
            "chunk_path": item["chunk_path"],
            "score": float(score)
        })

    return results


if __name__ == "__main__":

    query = input("Ask: ")

    for r in retrieve(query):

        print(r)