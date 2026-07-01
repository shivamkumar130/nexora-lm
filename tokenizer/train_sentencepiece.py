from pathlib import Path
import sentencepiece as spm

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "datasets" / "processed" / "training_corpus.txt"
MODEL_PREFIX = ROOT / "tokenizer" / "nexora_sp"

VOCAB_SIZE = 30000

def train_sentencepiece():
    spm.SentencePieceTrainer.train(
        input=str(CORPUS),
        model_prefix=str(MODEL_PREFIX),
        vocab_size=VOCAB_SIZE,
        model_type="bpe",
        character_coverage=1.0,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
    )
    print(f"Saved: {MODEL_PREFIX}.model")
    print(f"Saved: {MODEL_PREFIX}.vocab")

if __name__ == "__main__":
    train_sentencepiece()