import torch

from torch.utils.data import Dataset
from torch.utils.data import DataLoader


BLOCK_SIZE = 128
BATCH_SIZE = 32


class TextDataset(Dataset):

    def __init__(
        self,
        tokens,
        block_size
    ):

        self.tokens = tokens
        self.block_size = block_size


    def __len__(self):

        return (
            len(self.tokens)
            - self.block_size
        )


    def __getitem__(self, index):

        x = self.tokens[
            index:
            index + self.block_size
        ]

        y = self.tokens[
            index + 1:
            index + self.block_size + 1
        ]

        return x, y


def create_loaders(data):

    train_dataset = TextDataset(
        data["train"],
        BLOCK_SIZE
    )

    val_dataset = TextDataset(
        data["val"],
        BLOCK_SIZE
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return train_loader, val_loader