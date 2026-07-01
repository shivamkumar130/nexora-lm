from knowledge.search.hybrid_search import HybridSearch


class RetrieverEngine:
    def __init__(self):
        self.search = HybridSearch()

    def retrieve_context(self, query, top_k=8):
        return self.search.search(query, top_k=top_k)


if __name__ == "__main__":
    engine = RetrieverEngine()
    results = engine.retrieve_context("what is flexible sensor")
    print(results)