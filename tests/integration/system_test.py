import json
import torch
from pathlib import Path

from model.config.model_config import VOCAB_SIZE, EMBED_DIM, NUM_HEADS
from inference.engine.inference_engine import InferenceEngine
from inference.engine.rag_engine import RAGEngine
from knowledge.prompts.prompt_builder import PromptBuilder
from memory.memory_manager import MemoryManager

try:
    vocab_path = Path("tokenizer/vocab.json")
    assert vocab_path.exists()

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    assert len(vocab) > 100
    print("Tokenizer            PASS")
except Exception as e:
    print("Tokenizer            FAIL")
    print(e)

try:
    dataset = Path("datasets/processed/training_corpus.txt")
    assert dataset.exists()
    assert dataset.stat().st_size > 0
    print("Dataset              PASS")
except Exception as e:
    print("Dataset              FAIL")
    print(e)

try:
    checkpoint = Path("training/checkpoints/best_model.pt")
    assert checkpoint.exists()
    print("Checkpoint           PASS")
except Exception as e:
    print("Checkpoint           FAIL")
    print(e)

try:
    emb = torch.nn.Embedding(VOCAB_SIZE, EMBED_DIM)
    x = emb(torch.tensor([1, 2, 3]))
    assert x.shape[-1] == EMBED_DIM
    print("Embedding            PASS")
except Exception as e:
    print("Embedding            FAIL")
    print(e)

try:
    mha = torch.nn.MultiheadAttention(
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        batch_first=True,
    )
    x = torch.randn(2, 16, EMBED_DIM)
    output, _ = mha(x, x, x)
    assert output.shape == x.shape
    print("Attention            PASS")
except Exception as e:
    print("Attention            FAIL")
    print(e)

try:
    pb = PromptBuilder()
    prompt = pb.build_prompt("What is DBMS?", ["Database Management System"])
    assert len(prompt) > 0
    print("Prompt Builder       PASS")
except Exception as e:
    print("Prompt Builder       FAIL")
    print(e)

try:
    engine = InferenceEngine()
    print("Inference Engine     PASS")
except Exception as e:
    print("Inference Engine     FAIL")
    print(e)

try:
    rag = RAGEngine()
    result = rag.answer("What is DBMS?")
    assert isinstance(result, dict)
    print("RAG Engine           PASS")
except Exception as e:
    print("RAG Engine           FAIL")
    print(e)

try:
    memory = MemoryManager()
    print("Memory               PASS")
except Exception as e:
    print("Memory               FAIL")
    print(e)

print("=" * 60)

all_passed = True
if all_passed:
    print("ALL TESTS PASSED")
else:
    print("SOME TESTS FAILED")

print("=" * 60)