from inference.engine.inference_engine import InferenceEngine

from knowledge.search.hybrid_search import HybridSearch
from knowledge.ranking.reranker import Reranker
from knowledge.knowledge_manager import KnowledgeManager
from knowledge.prompts.prompt_builder import PromptBuilder
from knowledge.citations.source_tracker import SourceTracker
from knowledge.citations.citation_builder import CitationBuilder


class RAGEngine:

    def __init__(self):

        self.search = HybridSearch()
        self.reranker = Reranker()
        self.knowledge = KnowledgeManager()
        self.prompt_builder = PromptBuilder()
        self.generator = InferenceEngine()
        self.source_tracker = SourceTracker()
        self.citation_builder = CitationBuilder()

    def answer(self, question, top_k=5):

        # =====================================================
        # 1. Retrieve
        # =====================================================

        retrieved = self.search.search(
            question,
            top_k=top_k
        )

        if not retrieved:

            return {

                "question": question,

                "answer": "I don't know.",

                "contexts": [],

                "sources": [],

                "citations": "Sources:\nNone",

                "prompt": ""
            }

        # =====================================================
        # 2. Rerank
        # =====================================================

        ranked = self.reranker.rank(
            retrieved,
            top_k=3
        )

        # =====================================================
        # 3. Load Context
        # =====================================================

        contexts = self.knowledge.get_chunks(
            ranked
        )

        # =====================================================
        # 4. Build Prompt
        # =====================================================

        prompt = self.prompt_builder.build_prompt(
            question,
            contexts
        )

        # =====================================================
        # 5. Generate Answer
        # =====================================================

        answer = self.generator.generate(
            prompt=prompt,
            max_new_tokens=60,
            temperature=0.7,
            top_k=20
        ).strip()

        # =====================================================
        # 6. Track Sources
        # =====================================================

        sources = self.source_tracker.track_sources(
            ranked
        )

        citation_text = self.citation_builder.build(
            sources
        )

        # =====================================================
        # 7. Return
        # =====================================================

        return {

            "question": question,

            "answer": answer,

            "contexts": contexts,

            # Raw source list (used by benchmarks)
            "sources": sources,

            # Pretty formatted citations (used by CLI)
            "citations": citation_text,

            "prompt": prompt
        }


if __name__ == "__main__":

    rag = RAGEngine()

    while True:

        question = input("\nQuestion: ").strip()

        if question.lower() == "exit":
            break

        result = rag.answer(question)

        print("\n==============================")
        print("ANSWER")
        print("==============================")
        print(result["answer"])

        print("\n==============================")
        print("CITATIONS")
        print("==============================")
        print(result["citations"])