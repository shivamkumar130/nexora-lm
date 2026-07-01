import torch
import torch.nn as nn

from model.config.model_config import (
    EMBED_DIM,
    FFN_DIM,
    DROPOUT
)


class FeedForward(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(
                EMBED_DIM,
                FFN_DIM
            ),

            nn.GELU(),

            nn.Dropout(
                DROPOUT
            ),

            nn.Linear(
                FFN_DIM,
                EMBED_DIM
            ),

            nn.Dropout(
                DROPOUT
            )

        )

    def forward(self, x):

        return self.net(x)


if __name__ == "__main__":

    model = FeedForward()

    x = torch.randn(
        6,
        512,
        EMBED_DIM
    )

    output = model(x)

    print(output.shape)