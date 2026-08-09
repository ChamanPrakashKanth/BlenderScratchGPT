import json
import torch

INPUT_FILE = "data/blender_clean.txt"
OUTPUT_FILE = "data/tokenized.pt"


with open(
    INPUT_FILE,
    "r",
    encoding="utf-8"
) as f:

    text = f.read()


# --------------------------------
# Vocabulary
# --------------------------------

chars = sorted(set(text))

stoi = {
    ch: i
    for i, ch in enumerate(chars)
}

itos = {
    i: ch
    for ch, i in stoi.items()
}


# --------------------------------
# Text → token IDs
# --------------------------------

tokens = torch.tensor(
    [stoi[ch] for ch in text],
    dtype=torch.long
)


# --------------------------------
# Train / validation split
# --------------------------------

split = int(
    0.9 * len(tokens)
)

train_tokens = tokens[:split]
val_tokens = tokens[split:]


data = {
    "train": train_tokens,
    "val": val_tokens,
    "stoi": stoi,
    "itos": itos,
    "vocab_size": len(chars)
}


torch.save(
    data,
    OUTPUT_FILE
)


print("Vocabulary:", len(chars))
print("Total tokens:", len(tokens))
print("Training tokens:", len(train_tokens))
print("Validation tokens:", len(val_tokens))
print("Saved:", OUTPUT_FILE)