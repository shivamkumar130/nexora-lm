from knowledge.retriever.retriever import retrieve


class HybridSearch:

    def search(self, query, top_k=5):

        return retrieve(
            query=query,
            top_k=top_k
        )


if __name__ == "__main__":

    h = HybridSearch()

    print(
        h.search("what is dbms")
    )