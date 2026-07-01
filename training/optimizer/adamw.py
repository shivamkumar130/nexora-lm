import torch

from model.config.training_config import LEARNING_RATE, WEIGHT_DECAY

model = torch.nn.Linear(128, 128)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

if __name__ == "__main__":
    print("Optimizer Created")
    print(optimizer)