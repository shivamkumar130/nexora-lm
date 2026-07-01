from inference.engine.rag_engine import RAGEngine

print("=" * 60)
print("RAG CHAT TEST")
print("=" * 60)

rag = RAGEngine()

question = "What is Flexible Sensor?"

print("Question:", question)

chunks = rag.retriever.retrieve_context(question)

print("\nRetrieved Chunks:", len(chunks))

for chunk in chunks:
    print("-", chunk)

prompt = rag.prompt_builder.build(question, chunks)

print("\nPrompt Generated:", "YES")

response = rag.answer(question)

print("\nResponse Generated:", "YES")
print("\nResponse Preview:\n")
print(response["answer"][:500])

print("\n" + "=" * 60)
print("RAG CHAT TEST PASSED")
print("=" * 60)