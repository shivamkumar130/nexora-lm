from pathlib import Path
import torch
from torch.utils.data import Dataset

from tokenizer.tokenizer import encode

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "datasets" / "processed" / "training_corpus.txt"
CONTEXT_LENGTH = 512


class NexoraDataset(Dataset):

    def __init__(self):
        text = CORPUS.read_text(
            encoding="utf-8",
            errors="ignore"
        )

        self.tokens = encode(text)
        self.context_length = CONTEXT_LENGTH

    def __len__(self):
        return max(
            0,
            len(self.tokens) - self.context_length
        )

    def __getitem__(self, idx):
        chunk = self.tokens[idx:idx + self.context_length + 1]

        x = torch.tensor(
            chunk[:-1],
            dtype=torch.long
        )

        y = torch.tensor(
            chunk[1:],
            dtype=torch.long
        )

        return x, y


if __name__ == "__main__":

    import time

    start = time.time()

    dataset = NexoraDataset()

    print(f"Dataset loaded in {time.time()-start:.2f} sec")
    print("Dataset Samples:", len(dataset))

    x, y = dataset[0]

    print(x.shape)
    print(y.shape)