import torch

from dataset import create_loaders
from model import TinyGPT


DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


BLOCK_SIZE = 128
BATCH_SIZE = 32

N_EMBED = 128
N_HEAD = 4
N_LAYER = 4

LEARNING_RATE = 3e-4
STEPS = 2000

CHECKPOINT = (
    "checkpoints/blender_gpt.pt"
)


# -------------------------
# Load tokenizer/data
# -------------------------

data = torch.load(
    "data/tokenized.pt",
    weights_only=False
)

VOCAB_SIZE = data["vocab_size"]


# -------------------------
# Data
# -------------------------

train_loader, val_loader = (
    create_loaders(data)
)


# -------------------------
# Model
# -------------------------

model = TinyGPT(
    vocab_size=VOCAB_SIZE,
    block_size=BLOCK_SIZE,
    n_embed=N_EMBED,
    n_head=N_HEAD,
    n_layer=N_LAYER
)

model = model.to(DEVICE)


print(model)

print(
    "Parameters:",
    sum(
        p.numel()
        for p in model.parameters()
    )
)


# -------------------------
# Optimizer
# -------------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# -------------------------
# Training
# -------------------------

train_iter = iter(train_loader)

for step in range(1, STEPS + 1):

    try:

        x, y = next(train_iter)

    except StopIteration:

        train_iter = iter(train_loader)

        x, y = next(train_iter)


    x = x.to(DEVICE)
    y = y.to(DEVICE)

    optimizer.zero_grad()

    logits, loss = model(
        x,
        y
    )

    loss.backward()

    optimizer.step()


    if step == 1 or step % 50 == 0:

        print(
            f"Step {step:4d} | "
            f"Train Loss = "
            f"{loss.item():.4f}"
        )


# -------------------------
# Save
# -------------------------

torch.save(
    {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": {
            "vocab_size": VOCAB_SIZE,
            "block_size": BLOCK_SIZE,
            "n_embed": N_EMBED,
            "n_head": N_HEAD,
            "n_layer": N_LAYER
        }
    },
    CHECKPOINT
)

print("Model saved!")