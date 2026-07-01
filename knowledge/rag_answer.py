from inference.engine.rag_engine import RAGEngine


def main():
    rag = RAGEngine()
    while True:

        question = input("\nAsk: ").strip()

        if question.lower() == "exit":
            break

        if not question:
            continue

        data = rag.answer(question)

        print("\n=== Answer ===")
        print(data["answer"])

        print("\n=== Sources ===")
        print(data["citations"])
if __name__ == "__main__":
    main()