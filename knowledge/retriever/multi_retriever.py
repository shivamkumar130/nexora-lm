from knowledge.retriever.retriever import (
    retrieve
)


class MultiRetriever:
    def search(
        self,
        query,
        top_k=20
    ):
        results = retrieve(
            query,
            top_k=top_k
        )

        documents = {}

        for chunk in results:
            doc = chunk.get("folder_name")

            if not doc:
                chunk_id = chunk.get("chunk_id", "")
                doc = chunk_id.split("_chunk_")[0] if chunk_id else "unknown"

            if doc not in documents:
                documents[doc] = chunk

        return list(documents.values())


if __name__ == "__main__":
    r = MultiRetriever()

    print(
        r.search(
            "what is dbms"
        )
    )