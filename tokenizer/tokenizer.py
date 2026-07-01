from pathlib import Path
import sentencepiece as spm

ROOT = Path(__file__).resolve().parent
MODEL_FILE = ROOT / "nexora_sp.model"


class NexoraTokenizer:
    def __init__(self):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(str(MODEL_FILE))

    def tokenize(self, text):
        return self.sp.encode(text, out_type=str)

    def encode(self, text):
        return self.sp.encode(text, out_type=int)

    def decode(self, tokens):
        return self.sp.decode(tokens)


_TOKENIZER = NexoraTokenizer()

tokenize = _TOKENIZER.tokenize
encode = _TOKENIZER.encode
decode = _TOKENIZER.decode

if __name__ == "__main__":
    text = input("Text: ")
    ids = encode(text)
    print("Tokens:", tokenize(text))
    print("Ids:", ids)
    print("Decoded:", decode(ids))