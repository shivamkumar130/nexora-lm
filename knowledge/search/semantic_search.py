from pathlib import Path
from knowledge.retriever.retriever import retrieve
import json
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]

CHUNK_DIR = ROOT / "datasets" / "processed" / "chunks"

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / "search_logs.json"

while True:

    query = input("\nSearch > ")

    if query.lower() == "exit":
        break

    results = retrieve(query)

    print("\nRESULTS")
    print("=" * 50)

    # --------------------
    # Save Search History
    # --------------------
    log_entry = {
        "query": query,
        "time": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "results": results
    }

    logs = []

    if LOG_FILE.exists():

        try:

            with open(
                LOG_FILE,
                "r",
                encoding="utf-8"
            ) as f:

                logs = json.load(f)

        except:
            logs = []

    logs.append(log_entry)

    with open(
        LOG_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            logs,
            f,
            indent=4
        )

    # --------------------
    # Display Results
    # --------------------
    for result_name in results:

        try:

            parts = result_name.rsplit(
                "_chunk_",
                1
            )

            document_name = parts[0]
            chunk_number = parts[1]

            chunk_file = (
                CHUNK_DIR
                / document_name
                / f"chunk_{chunk_number}.txt"
            )

            if chunk_file.exists():

                text = chunk_file.read_text(
                    encoding="utf-8",
                    errors="ignore"
                )

                print("\n")
                print(result_name)

                print("-" * 50)

                print(text[:1000])

        except Exception as e:

            print("ERROR:", e)