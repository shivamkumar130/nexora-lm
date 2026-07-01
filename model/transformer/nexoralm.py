from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

import torch
import torch.nn as nn

from model.config.model_config import (
    VOCAB_SIZE,
    EMBED_DIM,
    NUM_HEADS,
    NUM_LAYERS,
    FFN_DIM,
    CONTEXT_LENGTH,
)


class NexoraLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.position_embedding = nn.Embedding(CONTEXT_LENGTH, EMBED_DIM)

        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=EMBED_DIM,
                    nhead=NUM_HEADS,
                    dim_feedforward=FFN_DIM,
                    dropout=0.1,
                    batch_first=True,
                    activation="gelu",
                    norm_first=True,
                )
                for _ in range(NUM_LAYERS)
            ]
        )

        self.ln_f = nn.LayerNorm(EMBED_DIM)
        self.lm_head = nn.Linear(EMBED_DIM, VOCAB_SIZE, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        b, t = idx.shape
        pos = torch.arange(t, device=idx.device).unsqueeze(0)
        x = self.token_embedding(idx) + self.position_embedding(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)


if __name__ == "__main__":
    import torch
    from model.config.model_config import BATCH_SIZE, CONTEXT_LENGTH, VOCAB_SIZE

    model = NexoraLM()
    x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, CONTEXT_LENGTH))
    y = model(x)

    print("Model loaded successfully")
    print("Input shape :", x.shape)
    print("Output shape:", y.shape)