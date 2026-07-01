import torch


def temperature_sample(
    logits,
    temperature=1.0
):
    temperature = max(temperature, 1e-5)

    logits = logits.float()
    logits = logits / temperature
    logits = logits - logits.max()
    probs = torch.softmax(
        logits,
        dim=-1
    )

    token = torch.multinomial(
        probs,
        num_samples=1
    )

    return token.item()


if __name__ == "__main__":
    x = torch.randn(5608)

    token = temperature_sample(
        x,
        temperature=0.7
    )

    print("Generated Token:")
    print(token)