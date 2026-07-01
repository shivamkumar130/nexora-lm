from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class KnowledgeManager:

    def get_chunk_text(self, chunk):

        chunk_path = ROOT / chunk["chunk_path"]

        if not chunk_path.exists():

            return f"[Missing Chunk: {chunk_path}]"

        return chunk_path.read_text(
            encoding="utf-8",
            errors="ignore"
        )

    def get_chunks(self, chunk_refs):

        contexts = []

        for chunk in chunk_refs:

            contexts.append(
                self.get_chunk_text(chunk)
            )

        return contexts