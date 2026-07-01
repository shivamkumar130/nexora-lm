from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable

steps = [
    "pdf_preprocess.py",
    "pdf_extractor.py",
    "text_cleaner.py",
    "metadata_generator.py",
    "chunk_generator.py",
    "embedding_generator.py",
    "build_training_corpus.py",
    "dataset_stats.py",
]

for step in steps:
    print("\n" + "=" * 60)
    print(f"RUNNING: {step}")
    print("=" * 60)

    step_path = ROOT / step
    if not step_path.exists():
        print(f"SKIPPING (missing): {step}")
        continue

    subprocess.run(
        [PYTHON, str(step_path)],
        check=True
    )

print("\nNEXORALM WEEK 2 PIPELINE COMPLETE")