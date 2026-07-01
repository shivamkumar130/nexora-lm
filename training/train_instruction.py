import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from training.data.instruction_dataset import InstructionDataset
from model.transformer.nexoralm import NexoraLM

# ==========================================================
# Paths
# ==========================================================

ROOT = Path(__file__).resolve().parents[1]

DATASET = ROOT / "datasets" / "instruction" / "instruction_train.json"

CHECKPOINT = Path(
    "/kaggle/input/datasets/skpaswan/checkpoints/best_model.pt"
)

SAVE_DIR = ROOT / "training" / "checkpoints"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

SAVE_MODEL = SAVE_DIR / "best_instruction_model.pt"

# ==========================================================
# Device
# ==========================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# ==========================================================
# Hyperparameters
# ==========================================================

BATCH_SIZE = 8
LEARNING_RATE = 5e-6
EPOCHS = 5
WEIGHT_DECAY = 0.01

# ==========================================================
# Dataset
# ==========================================================

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
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

# ==========================================================
# Model
# ==========================================================

model = NexoraLM().to(DEVICE)

# ==========================================================
# Load Pretrained Checkpoint
# ==========================================================

if not CHECKPOINT.exists():
    raise FileNotFoundError(
        f"Checkpoint not found:\n{CHECKPOINT}"
    )

print(f"Loading checkpoint:\n{CHECKPOINT}")

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
    weights_only=False,
)

model.load_state_dict(checkpoint["model_state_dict"])

print("Checkpoint loaded successfully.\n")

# ==========================================================
# Loss & Optimizer
# ==========================================================

criterion = nn.CrossEntropyLoss(ignore_index=0)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

# ==========================================================
# Validation
# ==========================================================

def evaluate():

    model.eval()

    loss_sum = 0.0

    with torch.no_grad():

        for x, y in val_loader:

            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            logits = model(x)

            loss = criterion(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
            )

            loss_sum += loss.item()

    return loss_sum / len(val_loader)

# ==========================================================
# Training
# ==========================================================

best_loss = float("inf")

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0.0

    for batch, (x, y) in enumerate(train_loader, start=1):

        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

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
                f"Epoch {epoch + 1}/{EPOCHS} | "
                f"Batch {batch}/{len(train_loader)} | "
                f"Loss: {loss.item():.4f}"
            )

    train_loss = total_loss / len(train_loader)

    val_loss = evaluate()

    perplexity = math.exp(val_loss)

    print("\n" + "=" * 60)

    print(f"Epoch        : {epoch + 1}")

    print(f"Train Loss   : {train_loss:.4f}")

    print(f"Val Loss     : {val_loss:.4f}")

    print(f"Perplexity   : {perplexity:.4f}")

    print("=" * 60)

    if val_loss < best_loss:

        best_loss = val_loss

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
            },
            SAVE_MODEL,
        )

        print(f"New best model saved to:\n{SAVE_MODEL}")

# ==========================================================
# Finished
# ==========================================================

print("\n" + "=" * 60)

print("Instruction Fine-Tuning Complete")

print(f"Best Validation Loss : {best_loss:.4f}")

print(f"Saved Model          : {SAVE_MODEL}")

print("=" * 60)