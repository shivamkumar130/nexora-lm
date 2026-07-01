import torch
import torch.nn as nn

from model.config.model_config import (
    CONTEXT_LENGTH,
    EMBED_DIM
)


class PositionEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(CONTEXT_LENGTH, EMBED_DIM)

    def forward(self, positions):
        if positions.max().item() >= CONTEXT_LENGTH:
            raise ValueError("Position index exceeds CONTEXT_LENGTH")
        return self.embedding(positions)


if __name__ == "__main__":
    model = PositionEmbedding()
    positions = torch.arange(CONTEXT_LENGTH, dtype=torch.long)
    output = model(positions)
    print("Position Embedding Shape:")
    print(output.shape)