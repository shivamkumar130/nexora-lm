from agents.base_agent import BaseAgent


class CodingAgent(BaseAgent):

    def __init__(self):

        super().__init__(
            "CodingAgent"
        )

    def run(self, query):

        return f"[Code] {query}"


if __name__ == "__main__":

    agent = CodingAgent()

    print(
        agent.run(
            "Write Bubble Sort"
        )
    )