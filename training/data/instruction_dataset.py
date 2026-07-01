import json

import torch
from torch.utils.data import Dataset

from tokenizer.tokenizer import encode

from model.config.model_config import CONTEXT_LENGTH


class InstructionDataset(Dataset):

    def __init__(self, json_file):

        with open(json_file, "r", encoding="utf-8") as f:

            self.samples = json.load(f)

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):

        sample = self.samples[idx]

        instruction = sample["instruction"].strip()

        user_input = sample.get("input", "").strip()

        output = sample["output"].strip()

        if user_input:

            prompt = (
                "### Instruction\n\n"
                f"{instruction}\n\n"
                "### Input\n\n"
                f"{user_input}\n\n"
                "### Response\n\n"
                f"{output}"
            )

        else:

            prompt = (
                "### Instruction\n\n"
                f"{instruction}\n\n"
                "### Response\n\n"
                f"{output}"
            )

        ids = encode(prompt)

        ids = ids[:CONTEXT_LENGTH]

        if len(ids) < CONTEXT_LENGTH:

            ids += [0] * (CONTEXT_LENGTH - len(ids))

        x = torch.tensor(ids[:-1], dtype=torch.long)

        y = torch.tensor(ids[1:], dtype=torch.long)

        return x, y