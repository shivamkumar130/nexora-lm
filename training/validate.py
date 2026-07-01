from training.data.dataset import NexoraDataset
from model.transformer.nexoralm import NexoraLM

import torch
import torch.nn as nn

dataset = NexoraDataset()
model = NexoraLM()

checkpoint = torch.load(
    "training/checkpoints/epoch_5.pt",
    map_location="cpu",
    weights_only=False,
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

loss_fn = nn.CrossEntropyLoss()
total_loss = 0.0

samples = min(100, len(dataset))

with torch.no_grad():
    for i in range(samples):
        x, y = dataset[i]

        x = torch.tensor(x, dtype=torch.long).unsqueeze(0)
        y = torch.tensor(y, dtype=torch.long).unsqueeze(0)

        logits = model(x)

        loss = loss_fn(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
        )

        total_loss += loss.item()

avg_loss = total_loss / max(1, samples)

print(f"Validation Loss: {avg_loss:.6f}")