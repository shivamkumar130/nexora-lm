from inference.engine.rag_engine import (
    RAGEngine
)

from memory.memory_manager import (
    MemoryManager
)

print("=" * 50)
print("NEXORAGPT CHAT")
print("=" * 50)

rag = RAGEngine()
memory = MemoryManager()

try:
    while True:
        query = input("\nYou: ")

        if not query.strip():
            continue

        if query.lower() == "exit":
            break

        try:

            response = rag.answer(query)

        except Exception as e:

            print(e)

            continue

        memory.save_exchange(

            query,

            {

                "answer": response["answer"],

                "sources": response["citations"]

            }

        )

        memory.save_working_memory({
            "query": query,
            "response": response["answer"]
        })

        print("\nNexoraGPT:\n")
        print(response["answer"])

        print("\nSources:")
        print(response["citations"])

except KeyboardInterrupt:
    print("\nBye!")