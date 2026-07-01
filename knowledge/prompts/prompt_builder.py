class PromptBuilder:

    def build_prompt(self, question, contexts):

        context_text = "\n\n".join(contexts)

        return f"""You are NexoraGPT.

Use ONLY the context below.

If the answer is not contained in the context, reply exactly:

I don't know.

Do not copy the context word-for-word.

Write a short answer in 2-4 sentences.

--------------------
CONTEXT
--------------------
{context_text}

--------------------
QUESTION
--------------------
{question}

--------------------
ANSWER
--------------------
"""
        return prompt

if __name__ == "__main__":
    builder = PromptBuilder()
    prompt = builder.build_prompt(
        "What is DBMS?",
        [
            "Database Management System stores data.",
            "It supports CRUD operations."
        ]
    )
    print(prompt)