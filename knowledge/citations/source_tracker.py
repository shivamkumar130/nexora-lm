class SourceTracker:

    def track_sources(self, ranked):

        sources = []

        for item in ranked:

            sources.append({

                "chunk_id": item["chunk_id"],

                "chunk_path": item["chunk_path"],

                "score": item.get("score", 0)

            })

        return sources