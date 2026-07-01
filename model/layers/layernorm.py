import torch
import torch.nn as nn

from model.config.model_config import EMBED_DIM


class LayerNorm(nn.Module):

    def __init__(self):
        super().__init__()

        self.norm = nn.LayerNorm(
            normalized_shape=EMBED_DIM
        )

    def forward(self, hidden_states):
        return self.norm(hidden_states)


if __name__ == "__main__":

    model = LayerNorm()

    x = torch.randn(
        6,
        512,
        EMBED_DIM
    )

    output = model(x)

    print("Input :", x.shape)
    print("Output:", output.shape)