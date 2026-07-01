import json
import math
import time
from collections import deque
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from training.data.dataset import NexoraDataset
from model.transformer.nexoralm import NexoraLM
from model.config.training_config import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    MAX_GRAD_NORM,
    SEED,
)


def save_log(log_file, entry):
    logs = []
    if log_file.exists():
        with open(log_file, "r", encoding="utf-8") as f:
            logs = json.load(f)

    logs.append(entry)

    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=4)


def train():
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("=" * 70)
    print("NexoraLM Training")
    print("=" * 70)
    print("Device:", device)

    root = Path(__file__).resolve().parent

    checkpoint_dir = root / "checkpoints"
    logs_dir = root / "logs"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_file = logs_dir / "training_logs.json"

    print("\nLoading dataset...")

    dataset = NexoraDataset()

    print("Dataset loaded successfully.")
    print(f"Total samples : {len(dataset):,}")

    train_size = int(len(dataset) * 0.8)
    valid_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(SEED)

    train_dataset, valid_dataset = random_split(
        dataset,
        [train_size, valid_size],
        generator=generator,
    )

    print(f"Training samples : {len(train_dataset):,}")
    print(f"Validation samples : {len(valid_dataset):,}")

    num_workers = 2 if torch.cuda.is_available() else 0

    print("\nCreating DataLoaders...")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )

    print("DataLoaders created.")
    print(f"Train batches : {len(train_loader):,}")
    print(f"Valid batches : {len(valid_loader):,}")

    print("\nBuilding model...")

    model = NexoraLM().to(device)

    print("Model moved to", device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
    )

    loss_fn = nn.CrossEntropyLoss()

    start_epoch = 0
    latest_checkpoint = checkpoint_dir / "latest.pt"

    if latest_checkpoint.exists():
        print("\nLoading latest checkpoint...")
        checkpoint = torch.load(latest_checkpoint, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        best_loss = checkpoint.get("best_loss", float("inf"))
        start_epoch = checkpoint["epoch"]

        print(f"Resuming from Epoch {start_epoch}")
    else:
        best_loss = float("inf")

    print("\nStarting training...")
    print("=" * 70)

    total_start = time.time()

    for epoch in range(start_epoch, EPOCHS):
        epoch_start = time.time()

        model.train()
        train_loss = 0.0

        batch_times = deque(maxlen=50)

        for batch_idx, (x, targets) in enumerate(train_loader):
            batch_start = time.time()

            x = x.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            logits = model(x)

            loss = loss_fn(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                MAX_GRAD_NORM,
            )

            optimizer.step()

            train_loss += loss.item()
            batch_times.append(time.time() - batch_start)

            if batch_idx == 0 or (batch_idx + 1) % 1000 == 0:
                avg_batch_time = sum(batch_times) / len(batch_times)
                remaining_batches = len(train_loader) - batch_idx - 1
                eta_seconds = int(avg_batch_time * remaining_batches)

                hours = eta_seconds // 3600
                minutes = (eta_seconds % 3600) // 60
                seconds = eta_seconds % 60

                eta_text = (
                    f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                    if hours > 0
                    else f"{minutes:02d}:{seconds:02d}"
                )

                print(
                    f"Epoch [{epoch + 1}/{EPOCHS}] "
                    f"Batch [{batch_idx + 1:,}/{len(train_loader):,}] "
                    f"Loss={loss.item():.4f} "
                    f"ETA={eta_text}",
                    flush=True,
                )

        scheduler.step()

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        valid_loss = 0.0

        with torch.no_grad():
            for batch_idx, (x, targets) in enumerate(valid_loader):
                if batch_idx % 500 == 0:
                    print(
                        f"Validation [{batch_idx:,}/{len(valid_loader):,}]",
                        flush=True,
                    )

                x = x.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                logits = model(x)

                loss = loss_fn(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                )

                valid_loss += loss.item()

        avg_valid_loss = valid_loss / len(valid_loader)
        perplexity = math.exp(min(avg_valid_loss, 20))
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start

        print("-" * 70)
        print(
            f"Epoch {epoch+1:02d}/{EPOCHS}"
            f"\nTrain Loss : {avg_train_loss:.4f}"
            f"\nValid Loss : {avg_valid_loss:.4f}"
            f"\nPerplexity : {perplexity:.2f}"
            f"\nLearning Rate : {current_lr:.8f}"
            f"\nEpoch Time : {epoch_time/60:.2f} minutes"
        )
        print("-" * 70)

        if avg_valid_loss < best_loss:
            best_loss = avg_valid_loss

            best_checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_loss": avg_train_loss,
                "valid_loss": avg_valid_loss,
                "perplexity": perplexity,
                "best_loss": best_loss,
            }

            torch.save(
                best_checkpoint,
                checkpoint_dir / "best_model.pt",
            )

            print("✓ Best Model Saved")

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": avg_train_loss,
            "valid_loss": avg_valid_loss,
            "perplexity": perplexity,
            "best_loss": best_loss,
        }

        torch.save(
            checkpoint,
            checkpoint_dir / "latest.pt",
        )

        torch.save(
            checkpoint,
            checkpoint_dir / f"epoch_{epoch+1}.pt",
        )

        print("Checkpoint Saved.")

        save_log(
            log_file,
            {
                "epoch": epoch + 1,
                "train_loss": round(avg_train_loss, 6),
                "valid_loss": round(avg_valid_loss, 6),
                "perplexity": round(perplexity, 6),
                "learning_rate": current_lr,
            },
        )

    total_time = time.time() - total_start

    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best Validation Loss : {best_loss:.6f}")
    print(f"Total Training Time : {total_time/3600:.2f} hours")
    print("=" * 70)


if __name__ == "__main__":
    train()