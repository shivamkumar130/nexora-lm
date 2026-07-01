import torch


def greedy_sample(logits):
    if logits.ndim != 1:
        logits = logits.squeeze()

    return torch.argmax(
        logits,
        dim=-1
    ).item()


if __name__ == "__main__":
    x = torch.randn(5608)

    print(
        "Generated Token:",
        greedy_sample(x)
    )