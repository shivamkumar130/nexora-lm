import torch

from tokenizer.tokenizer import encode, decode
from model.transformer.nexoralm import NexoraLM
from model.config.model_config import CONTEXT_LENGTH, EOS_ID


class InferenceEngine:
    def __init__(self, checkpoint_path="training/checkpoints/best_instruction_model.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = NexoraLM().to(self.device)

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    @torch.no_grad()
    def generate(
        self,
        prompt,
        max_new_tokens=50,
        temperature=0.8,
        top_k=20,
        repetition_penalty=1.1,
    ):
        tokens = encode(prompt)

        if len(tokens) > CONTEXT_LENGTH:
            tokens = tokens[-CONTEXT_LENGTH:]

        context = torch.tensor([tokens], dtype=torch.long, device=self.device)
        generated_ids = []

        for _ in range(max_new_tokens):
            logits = self.model(context)
            logits = logits[:, -1, :]

            logits = logits / max(temperature, 1e-8)

            if repetition_penalty > 1.0:
                for token_id in set(context[0].tolist()):
                    logits[0, token_id] /= repetition_penalty

            if top_k is not None:
                k = min(top_k, logits.size(-1))
                values, _ = torch.topk(logits, k)
                logits[logits < values[:, [-1]]] = -float("inf")

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            next_id = next_token.item()

            generated_ids.append(next_id)

            if next_id == EOS_ID:
                break

            context = torch.cat([context, next_token], dim=1)

            if context.size(1) > CONTEXT_LENGTH:
                context = context[:, -CONTEXT_LENGTH:]

        return decode(generated_ids).strip()


if __name__ == "__main__":
    engine = InferenceEngine()

    while True:
        prompt = input("\nPrompt: ")

        if prompt.lower() == "exit":
            break

        print()
        print(
            engine.generate(
                prompt,
                max_new_tokens=60,
                temperature=0.8,
                top_k=20,
                repetition_penalty=1.1,
            )
        )