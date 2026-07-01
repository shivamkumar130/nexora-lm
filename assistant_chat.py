from inference.engine.assistant_engine import AssistantEngine

assistant = AssistantEngine()

print("=" * 60)
print("NEXORA ASSISTANT")
print("=" * 60)

while True:

    question = input("\nYou : ")

    if question.lower() == "exit":
        break

    answer = assistant.generate(question)

    print("\nNexora :")
    print(answer)