import torch
import torch.nn as nn

from model.attention.multihead_attention import MultiHeadAttention
from model.layers.feedforward import FeedForward
from model.layers.layernorm import LayerNorm


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.norm1 = LayerNorm()
        self.attention = MultiHeadAttention()
        self.norm2 = LayerNorm()
        self.feedforward = FeedForward()

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.feedforward(self.norm2(x))
        return x


if __name__ == "__main__":
    model = TransformerBlock()

    x = torch.randn(6, 512, 256)
    output = model(x)

    print(output.shape)