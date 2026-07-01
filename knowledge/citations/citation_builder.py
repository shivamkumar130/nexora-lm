class CitationBuilder:

    def build(self, sources):

        if not sources:

            return "\nSources:\nNone"

        lines = ["\nSources:"]

        for i, src in enumerate(sources, start=1):

            lines.append(

                f"[{i}] {src['chunk_id']}"

                f" | Score={src['score']:.4f}"

            )

        return "\n".join(lines)