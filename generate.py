import torch

from model import TinyGPT


DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# -------------------------
# Load data/tokenizer
# -------------------------

data = torch.load(
    "data/tokenized.pt",
    weights_only=False
)

stoi = data["stoi"]
itos = data["itos"]


# -------------------------
# Rebuild model
# -------------------------

checkpoint = torch.load(
    "checkpoints/blender_gpt.pt",
    map_location=DEVICE,
    weights_only=False
)

config = checkpoint["config"]


model = TinyGPT(
    vocab_size=config["vocab_size"],
    block_size=config["block_size"],
    n_embed=config["n_embed"],
    n_head=config["n_head"],
    n_layer=config["n_layer"]
)

model.load_state_dict(
    checkpoint["model"]
)

model.to(DEVICE)
model.eval()


# -------------------------
# Encode
# -------------------------

def encode(text):

    return [
        stoi[c]
        for c in text
        if c in stoi
    ]


# -------------------------
# Decode
# -------------------------

def decode(tokens):

    return "".join(
        itos[int(t)]
        for t in tokens
    )


# -------------------------
# Generate
# -------------------------

@torch.no_grad()
def generate(
    prompt,
    max_new_tokens=200,
    temperature=1.0
):

    ids = encode(prompt)

    x = torch.tensor(
        [ids],
        dtype=torch.long,
        device=DEVICE
    )

    for _ in range(max_new_tokens):

        x_cond = x[
            :, -config["block_size"]:
        ]

        logits, _ = model(x_cond)

        logits = logits[:, -1, :]

        logits = logits / temperature

        probs = torch.softmax(
            logits,
            dim=-1
        )

        next_token = torch.multinomial(
            probs,
            num_samples=1
        )

        x = torch.cat(
            [x, next_token],
            dim=1
        )

    return decode(
        x[0].tolist()
    )


prompt = input("Prompt: ")

print(
    generate(prompt)
)