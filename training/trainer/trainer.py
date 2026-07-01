import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from training.data.dataset import NexoraDataset
from model.transformer.nexoralm import NexoraLM
from model.config.training_config import BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY

dataset = NexoraDataset()

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=torch.cuda.is_available(),
)

model = NexoraLM()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

loss_fn = nn.CrossEntropyLoss()

model.train()

for x, targets in loader:
    logits = model(x)

    loss = loss_fn(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("Loss:", loss.item())
    break