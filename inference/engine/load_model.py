from pathlib import Path
import torch

CHECKPOINT_DIR = (
    Path(__file__).resolve().parents[2]
    / "training"
    / "checkpoints"
)

best_model = CHECKPOINT_DIR / "best_model.pt"

if best_model.exists():
    checkpoint = torch.load(
        best_model,
        map_location="cpu",
        weights_only=False
    )

    loss = checkpoint.get(
        "best_loss",
        checkpoint.get(
            "valid_loss",
            "Unknown"
        )
    )

    print("Best Model Loaded")
    print(best_model.name)
    print("Epoch:", checkpoint.get("epoch", "Unknown"))
    print("Validation Loss:", loss)
else:
    print("Best Model Not Found")