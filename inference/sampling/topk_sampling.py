import torch


def topk_sample(
    logits,
    k=10
):
    k = max(
        1,
        min(k, logits.size(-1))
    )

    values, indices = torch.topk(
        logits,
        k
    )

    probs = torch.softmax(
        values,
        dim=-1
    )

    selected = torch.multinomial(
        probs,
        num_samples=1
    )

    return indices[selected].item()


if __name__ == "__main__":
    x = torch.randn(5608)

    token = topk_sample(
        x,
        k=10
    )

    print("Generated Token:")
    print(token)