import torch
import torch.nn as nn

from model.config.model_config import (
    EMBED_DIM,
    NUM_HEADS,
    DROPOUT
)


class MultiHeadAttention(nn.Module):

    def __init__(self):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS,
            dropout=DROPOUT,
            batch_first=True
        )

    def forward(self, x):

        output, attention_weights = self.attention(
            x,
            x,
            x,
            need_weights=False
        )

        return output


if __name__ == "__main__":

    model = MultiHeadAttention()

    x = torch.randn(
        6,
        512,
        EMBED_DIM
    )

    output = model(x)

    print("Input Shape :", x.shape)
    print("Output Shape:", output.shape)