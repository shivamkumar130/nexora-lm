import torch
import torch.nn as nn

from model.config.model_config import (
    VOCAB_SIZE,
    EMBED_DIM
)


class TokenEmbedding(nn.Module):

    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=VOCAB_SIZE,
            embedding_dim=EMBED_DIM
        )

    def forward(self, tokens):

        return self.embedding(tokens)


if __name__ == "__main__":

    model = TokenEmbedding()

    tokens = torch.tensor([[15, 92, 341]])

    output = model(tokens)

    print("Input Shape :", tokens.shape)
    print("Output Shape:", output.shape)