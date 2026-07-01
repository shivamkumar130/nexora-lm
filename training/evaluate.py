import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.data.dataset import NexoraDataset
from model.transformer.nexoralm import NexoraLM
from model.config.model_config import BATCH_SIZE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = NexoraDataset()
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=torch.cuda.is_available(),
)

model = NexoraLM().to(device)

checkpoint = torch.load(
    "training/checkpoints/best_model.pt",
    map_location=device,
    weights_only=False,
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

loss_fn = nn.CrossEntropyLoss()
total_loss = 0.0

with torch.no_grad():
    for x, targets in loader:
        x = x.to(device)
        targets = targets.to(device)

        logits = model(x)
        loss = loss_fn(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
        )
        total_loss += loss.item()

avg_loss = total_loss / max(1, len(loader))
perplexity = math.exp(min(avg_loss, 20))

print("=" * 50)
print("NEXORALM EVALUATION")
print("=" * 50)
print("Device:", device)
print("Loss:", round(avg_loss, 6))
print("Perplexity:", round(perplexity, 6))
print("=" * 50)