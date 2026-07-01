from pathlib import Path


def dataset_stats(folder):

    if not folder.exists():
        raise FileNotFoundError(folder)

    files = sorted(folder.rglob("*.txt"))

    total_files = len(files)
    total_words = 0
    total_lines = 0
    total_characters = 0

    for f in files:

        try:
            text = f.read_text(
                encoding="utf-8",
                errors="ignore"
            )

            total_words += len(text.split())
            total_lines += len(text.splitlines())
            total_characters += len(text)

        except Exception as e:
            print(f"ERROR: {f}")
            print(e)

    print("\nDataset Summary")
    print("=" * 40)
    print(f"Files      : {total_files}")
    print(f"Words      : {total_words:,}")
    print(f"Lines      : {total_lines:,}")
    print(f"Characters : {total_characters:,}")


if __name__ == "__main__":

    ROOT = Path(__file__).resolve().parent.parent

    CLEAN_DIR = ROOT / "processed" / "clean_text"

    print(f"Scanning: {CLEAN_DIR}")

    dataset_stats(CLEAN_DIR)