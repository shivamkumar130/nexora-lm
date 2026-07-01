from agents.base_agent import BaseAgent


class KnowledgeAgent(BaseAgent):

    def __init__(self):

        super().__init__(
            "KnowledgeAgent"
        )

    def run(self, query):

        return f"[Knowledge] {query}"


if __name__ == "__main__":

    agent = KnowledgeAgent()

    print(
        agent.run(
            "What is DBMS?"
        )
    )