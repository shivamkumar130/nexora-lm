import torch
import torch.nn as nn

loss_fn = nn.CrossEntropyLoss()

if __name__ == "__main__":
    logits = torch.randn(8, 5608)
    targets = torch.randint(0, 5608, (8,))
    loss = loss_fn(logits, targets)
    print("Loss:", loss.item())