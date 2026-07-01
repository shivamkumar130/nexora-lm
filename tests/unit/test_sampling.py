import torch

from inference.sampling.greedy_decode import (
    greedy_sample
)

x = torch.randn(5608)

token = greedy_sample(x)

print(
    "Greedy Sampling PASS"
)

print(
    "Token:",
    token
)