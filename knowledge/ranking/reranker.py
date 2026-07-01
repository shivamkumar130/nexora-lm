class Reranker:
    def rank(self, retrieved, top_k=3):
        if not retrieved:
            return []
        return sorted(
            retrieved,
            key=lambda x: x.get("score", 0),
            reverse=True
        )[:top_k]


if __name__ == "__main__":
    r = Reranker()
    sample = [
        {"chunk_id": "1", "text": "A", "score": 0.3},
        {"chunk_id": "2", "text": "B", "score": 0.9},
    ]
    print(r.rank(sample))