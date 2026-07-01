import torch

from model.transformer.nexoralm import NexoraLM
from tokenizer.tokenizer import encode, decode
from model.config.model_config import CONTEXT_LENGTH

CHECKPOINT = "training/checkpoints/best_model.pt"
DEVICE = "cpu"
MAX_NEW_TOKENS = 50


def load_model():
    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=False
    )

    model = NexoraLM()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    return model


model = load_model()


def generate(prompt):
    tokens = encode(prompt)

    if len(tokens) == 0:
        return ""

    context = torch.tensor([tokens], dtype=torch.long, device=DEVICE)

    if context.size(1) == 0:
        return ""

    with torch.no_grad():
        for _ in range(MAX_NEW_TOKENS):
            if context.size(1) > CONTEXT_LENGTH:
                context = context[:, -CONTEXT_LENGTH:]

            logits = model(context)
            next_token = torch.argmax(logits[:, -1, :], dim=-1)

            context = torch.cat(
                [context, next_token.unsqueeze(1)],
                dim=1
            )

            if context.size(1) > CONTEXT_LENGTH:
                context = context[:, -CONTEXT_LENGTH:]

    output_tokens = context[0].tolist()
    return decode(output_tokens)


if __name__ == "__main__":
    prompt = input("\nPrompt: ")
    result = generate(prompt)

    print("\nGenerated Text:\n")
    print(result)