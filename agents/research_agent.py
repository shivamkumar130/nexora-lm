from agents.base_agent import BaseAgent


class ResearchAgent(BaseAgent):

    def __init__(self):

        super().__init__(
            "ResearchAgent"
        )

    def run(self, query):

        return f"[Research] {query}"


if __name__ == "__main__":

    agent = ResearchAgent()

    print(
        agent.run(
            "Explain AI"
        )
    )