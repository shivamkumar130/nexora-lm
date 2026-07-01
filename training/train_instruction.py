import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from training.data.instruction_dataset import InstructionDataset
from model.transformer.nexoralm import NexoraLM

ROOT = Path(__file__).resolve().parents[1]

DATASET = ROOT / "datasets" / "instruction" / "instruction_train.json"

CHECKPOINT = ROOT / "training" / "checkpoints" / "best_model.pt"

SAVE_DIR = ROOT / "training" / "checkpoints"
SAVE_DIR.mkdir(exist_ok=True)

SAVE_MODEL = SAVE_DIR / "best_instruction_model.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 8
LEARNING_RATE = 5e-6
EPOCHS = 5
WEIGHT_DECAY = 0.01

dataset = InstructionDataset(DATASET)

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
)

model = NexoraLM().to(DEVICE)

from pathlib import Path

CHECKPOINT = Path("/kaggle/input/datasets/skpaswan/checkpoints/best_model.pt")

checkpoint = torch.load(
    CHECKPOINT,
    map_location=torch.device,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

print("Loaded:", CHECKPOINT)

criterion = nn.CrossEntropyLoss(ignore_index=0)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

def evaluate():

    model.eval()

    loss_sum = 0

    with torch.no_grad():

        for x, y in val_loader:

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(x)

            loss = criterion(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
            )

            loss_sum += loss.item()

    return loss_sum / len(val_loader)


best_loss = float("inf")

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    for batch, (x, y) in enumerate(train_loader, start=1):

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()

        logits = model(x)

        loss = criterion(
            logits.view(-1, logits.size(-1)),
            y.view(-1),
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        if batch % 10 == 0:

            print(
                f"Epoch {epoch+1} "
                f"Batch {batch} "
                f"Loss {loss.item():.4f}"
            )

    train_loss = total_loss / len(train_loader)

    val_loss = evaluate()

    perplexity = math.exp(val_loss)

    print()

    print("=" * 60)

    print(f"Epoch {epoch+1}")

    print(f"Train Loss : {train_loss:.4f}")

    print(f"Val Loss   : {val_loss:.4f}")

    print(f"Perplexity : {perplexity:.4f}")

    print("=" * 60)

    if val_loss < best_loss:

        best_loss = val_loss

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
            },
            SAVE_MODEL,
        )

        print("Saved:", SAVE_MODEL)


print()

print("=" * 60)

print("Instruction Fine-Tuning Complete")

print("Best Validation Loss :", best_loss)

print("Saved Model :", SAVE_MODEL)

print("=" * 60)