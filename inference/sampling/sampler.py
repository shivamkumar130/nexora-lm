from inference.sampling.greedy_decode import (
    greedy_sample
)

from inference.sampling.temperature import (
    temperature_sample
)

from inference.sampling.topk_sampling import (
    topk_sample
)


class Sampler:
    @staticmethod
    def sample(
        logits,
        method="topk"
    ):
        if method == "greedy":
            return greedy_sample(logits)

        elif method == "temperature":
            return temperature_sample(
                logits,
                temperature=0.7
            )

        elif method == "topk":
            return topk_sample(
                logits,
                k=10
            )

        else:
            raise ValueError(
                "Unknown Sampling Method"
            )