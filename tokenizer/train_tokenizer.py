from pathlib import Path
import json

from tokenizer.tokenizer import NexoraTokenizer

ROOT = Path(__file__).resolve().parent.parent
VOCAB_FILE = ROOT / "tokenizer" / "vocab.json"

tokenizer = NexoraTokenizer()

vocab = {}
size = tokenizer.sp.get_piece_size()

for i in range(size):
    piece = tokenizer.sp.id_to_piece(i)
    vocab[piece] = i

with open(VOCAB_FILE, "w", encoding="utf-8") as f:
    json.dump(vocab, f, indent=4, ensure_ascii=False)

print("Vocabulary Created")
print("Tokens:", len(vocab))