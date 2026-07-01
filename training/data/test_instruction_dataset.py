from pathlib import Path

from training.data.instruction_dataset import InstructionDataset

ROOT = Path(__file__).resolve().parents[2]

dataset = InstructionDataset(

    ROOT /
    "datasets" /
    "instruction" /
    "instruction_train.json"

)

print("=" * 60)

print("Instruction Dataset")

print("=" * 60)

print("Samples :", len(dataset))
sample = dataset.samples[0]

print("\nInstruction")
print(sample["instruction"])

print("\nInput")
print(sample.get("input", ""))

print("\nOutput")
print(sample["output"])
x, y = dataset[0]

print()

print("Input Shape :", x.shape)

print("Target Shape:", y.shape)

print()

print(x[:40])

print()

print(y[:40])