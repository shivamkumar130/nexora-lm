from pathlib import Path
from tokenizer.tokenizer import tokenize

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "datasets" / "processed" / "training_corpus.txt"
TRAIN = ROOT / "training" / "data" / "train_split.txt"
VALID = ROOT / "training" / "data" / "valid_split.txt"

text = CORPUS.read_text(encoding="utf-8", errors="ignore")
tokens = tokenize(text)

split_index = int(len(tokens) * 0.8)

train_tokens = tokens[:split_index]
valid_tokens = tokens[split_index:]

TRAIN.write_text(" ".join(train_tokens), encoding="utf-8")
VALID.write_text(" ".join(valid_tokens), encoding="utf-8")

print("Train Tokens:", len(train_tokens))
print("Valid Tokens:", len(valid_tokens))