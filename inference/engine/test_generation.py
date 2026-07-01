from pathlib import Path
import torch

from model.transformer.nexoralm import NexoraLM

ROOT = Path(__file__).resolve().parents[1]
checkpoint_path = ROOT / "training" / "checkpoints" / "best_model.pt"

checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=False
)

model = NexoraLM()

if "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

params = sum(p.numel() for p in model.parameters())
loss_value = checkpoint.get("best_loss", checkpoint.get("valid_loss", "Unknown"))

print("Model Loaded")
print(f"Parameters: {params:,}")
print("Best Validation Loss:", loss_value)