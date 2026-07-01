from pathlib import Path
import json

from tokenizer.tokenizer import tokenize

ROOT = Path(__file__).resolve().parents[2]
VOCAB = ROOT / "tokenizer" / "vocab.json"
CORPUS = ROOT / "datasets" / "processed" / "training_corpus.txt"

with open(VOCAB, "r", encoding="utf-8") as f:
    vocab = json.load(f)

text = CORPUS.read_text(encoding="utf-8", errors="ignore")
tokens = tokenize(text)

print("=" * 50)
print("NEXORALM BENCHMARK")
print("=" * 50)
print("Vocabulary Size :", len(vocab))
print("Corpus Tokens   :", len(tokens))
print("Corpus Chars    :", len(text))
print("=" * 50)