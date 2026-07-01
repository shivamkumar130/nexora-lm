from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]

sys.path.append(str(ROOT))

import torch

from model.embedding.token_embedding import TokenEmbedding
from model.embedding.position_embedding import PositionEmbedding

from model.attention.self_attention import SelfAttention
from model.attention.multihead_attention import MultiHeadAttention

from model.layers.feedforward import FeedForward
from model.layers.layernorm import LayerNorm
from model.layers.dropout import DropoutLayer
from model.layers.residual import ResidualConnection

from model.transformer.transformer_block import TransformerBlock
from model.transformer.nexoralm import NexoraLM

print("=" * 60)
print("WEEK 4 VALIDATION")
print("=" * 60)

# --------------------------------------------------
# Token Embedding
# --------------------------------------------------

tokens = torch.tensor([1, 2, 3])

layer = TokenEmbedding()

output = layer(tokens)

print("Token Embedding      PASS", output.shape)

# --------------------------------------------------
# Position Embedding
# --------------------------------------------------

print("Testing Position Embedding...")

pos = PositionEmbedding()

output = pos(128)

print(
    "Position Embedding",
    "PASS",
    output.shape
)
# --------------------------------------------------
# Self Attention
# --------------------------------------------------

x = torch.randn(2, 128, 128)

attn = SelfAttention()

output = attn(x)

print("Self Attention       PASS", output.shape)

# --------------------------------------------------
# Multi Head
# --------------------------------------------------

mha = MultiHeadAttention()

output = mha(x)

print("Multi Head           PASS", output.shape)

# --------------------------------------------------
# Feed Forward
# --------------------------------------------------

ffn = FeedForward()

output = ffn(x)

print("Feed Forward         PASS", output.shape)

# --------------------------------------------------
# LayerNorm
# --------------------------------------------------

norm = LayerNorm()

output = norm(x)

print("LayerNorm            PASS", output.shape)

# --------------------------------------------------
# Dropout
# --------------------------------------------------

drop = DropoutLayer()

output = drop(x)

print("Dropout              PASS", output.shape)

# --------------------------------------------------
# Residual
# --------------------------------------------------

res = ResidualConnection()

output = res.add(x, x)

print(
    "Residual",
    "PASS",
    output.shape
)
# --------------------------------------------------
# Transformer Block
# --------------------------------------------------

block = TransformerBlock()

output = block(x)

print("Transformer Block    PASS", output.shape)

# --------------------------------------------------
# NexoraLM
# --------------------------------------------------

model = NexoraLM()

tokens = torch.randint(
    0,
    100,
    (2, 128)
)

output = model(tokens)

print("NexoraLM PASS", output.shape)

print("\n")
print("=" * 60)
print("WEEK 4 COMPLETED SUCCESSFULLY")
print("=" * 60)