import json
from pathlib import Path


class MemoryManager:
    def __init__(self):
        self.root = Path(__file__).resolve().parent

        self.chat_history = self.root / "chat_history.json"
        self.short_term = self.root / "short_term" / "short_term_memory.json"
        self.long_term = self.root / "long_term" / "long_term_memory.json"
        self.working = self.root / "working" / "working_memory.json"

        self._initialize()

    def _initialize(self):
        files = [
            self.chat_history,
            self.short_term,
            self.long_term,
            self.working
        ]

        for file in files:
            file.parent.mkdir(parents=True, exist_ok=True)
            if not file.exists():
                file.write_text("[]", encoding="utf-8")

    def _load(self, path):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def _save(self, path, data):
        path.write_text(
            json.dumps(data, indent=4, ensure_ascii=False),
            encoding="utf-8"
        )

    def save_exchange(self, query, response):
        history = self._load(self.chat_history)
        history.append({
            "query": query,
            "response": response
        })
        history = history[-5:]
        self._save(self.chat_history, history)

    def get_recent_history(self):
        return self._load(self.chat_history)

    def save_working_memory(self, item):
        data = self._load(self.working)
        data.append(item)
        data = data[-20:]
        self._save(self.working, data)

    def save_short_term(self, item):
        data = self._load(self.short_term)
        data.append(item)
        data = data[-20:]
        self._save(self.short_term, data)

    def save_long_term(self, item):
        data = self._load(self.long_term)
        data.append(item)
        self._save(self.long_term, data)

    def summarize_session(self):
        history = self.get_recent_history()
        return {
            "messages": len(history),
            "topics": [x.get("query", "") for x in history]
        }

    def get_topic_summary(self):
        history = self.get_recent_history()
        topics = []

        for item in history:
            query = item.get("query", "")
            words = query.split()
            if len(words) >= 2:
                topics.append(" ".join(words[:2]))

        return topics


if __name__ == "__main__":
    memory = MemoryManager()
    print(memory.summarize_session())
    print()
    print(memory.get_topic_summary())