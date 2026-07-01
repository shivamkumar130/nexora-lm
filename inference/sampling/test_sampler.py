import torch


from inference.sampling.sampler import (
    Sampler
)

x = torch.randn(5608)

print(
    "Greedy:",
    Sampler.sample(
        x,
        "greedy"
    )
)

print(
    Sampler.sample(
        x,
        "temperature"
    )
)

print(
    Sampler.sample(
        x,
        "topk"
    )
)