import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from model.config.training_config import EPOCHS, LEARNING_RATE

model = torch.nn.Linear(512, 256)

optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
)

scheduler = CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS,
)

print("Scheduler Created\n")

for epoch in range(EPOCHS):
    optimizer.step()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scheduler.step()

    lr = optimizer.param_groups[0]["lr"]
    print(f"Epoch {epoch+1} | LR = {lr:.8f}")