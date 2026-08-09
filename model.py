import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierPositionalEncoding(nn.Module):

    def __init__(self,block_size,n_embed):

        super().__init__()

        pe = torch.zeros(block_size,n_embed)               

        position = torch.arange(
            block_size,
            dtype=torch.float32
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                n_embed,
                2
            ).float()
            *
            (
                -math.log(10000.0)
                / n_embed
            )
        )

        pe[:, 0::2] = torch.sin(
            position * div_term
        )

        pe[:, 1::2] = torch.cos(
            position * div_term
        )

        self.register_buffer(
            "pe",
            pe
        )


    def forward(self, x):

        B, T, C = x.shape

        return (
            x
            +
            self.pe[:T].unsqueeze(0)
        )

class ComplexReLU(nn.Module):

    def forward(
        self,
        real,
        imag
    ):

        real = torch.relu(real)
        imag = torch.relu(imag)

        return real, imag
class ComplexFeedForward(
    nn.Module
):

    def __init__(
        self,
        n_embed
    ):

        super().__init__()

        self.fc1 = nn.Linear(
            n_embed,
            4 * n_embed
        )

        self.activation = ComplexReLU()

        self.fc2 = nn.Linear(
            4 * n_embed,
            n_embed
        )


    def forward(self, x):

        z = self.fc1(x)

        real, imag = torch.chunk(
            z,
            2,
            dim=-1
        )

        real, imag = self.activation(
            real,
            imag
        )

        z = torch.cat(
            [real, imag],
            dim=-1
        )

        return self.fc2(z)
class SelfAttention(nn.Module):

    def __init__(
        self,
        n_embed,
        n_head,
        block_size
    ):

        super().__init__()

        assert n_embed % n_head == 0

        self.n_head = n_head

        self.head_dim = (
            n_embed // n_head
        )

        self.key = nn.Linear(
            n_embed,
            n_embed
        )

        self.query = nn.Linear(
            n_embed,
            n_embed
        )

        self.value = nn.Linear(
            n_embed,
            n_embed
        )

        self.proj = nn.Linear(
            n_embed,
            n_embed
        )

        self.register_buffer(
            "mask",
            torch.tril(
                torch.ones(
                    block_size,
                    block_size
                )
            )
        )


    def forward(self, x):

        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        k = k.view(
            B,
            T,
            self.n_head,
            self.head_dim
        ).transpose(1, 2)

        q = q.view(
            B,
            T,
            self.n_head,
            self.head_dim
        ).transpose(1, 2)

        v = v.view(
            B,
            T,
            self.n_head,
            self.head_dim
        ).transpose(1, 2)

        scores = (
            q
            @
            k.transpose(-2, -1)
        )

        scores = scores / math.sqrt(
            self.head_dim
        )

        scores = scores.masked_fill(
            self.mask[:T, :T] == 0,
            float("-inf")
        )

        attention = F.softmax(
            scores,
            dim=-1
        )

        out = attention @ v

        out = out.transpose(
            1,
            2
        ).contiguous()

        out = out.view(
            B,
            T,
            C
        )

        return self.proj(out)
class Block(nn.Module):

    def __init__(
        self,
        n_embed,
        n_head,
        block_size
    ):

        super().__init__()

        self.ln1 = nn.LayerNorm(
            n_embed
        )

        self.attention = SelfAttention(
            n_embed,
            n_head,
            block_size
        )

        self.ln2 = nn.LayerNorm(
            n_embed
        )

        self.ff = ComplexFeedForward(
            n_embed
        )


    def forward(self, x):

        x = x + self.attention(
            self.ln1(x)
        )

        x = x + self.ff(
            self.ln2(x)
        )

        return x
class TinyGPT(nn.Module):

    def __init__(
        self,
        vocab_size,
        block_size=128,
        n_embed=128,
        n_head=4,
        n_layer=4
    ):

        super().__init__()

        self.block_size = block_size

        self.token_embedding = nn.Embedding(
            vocab_size,
            n_embed
        )

        self.position_encoding = (
            FourierPositionalEncoding(
                block_size,
                n_embed
            )
        )

        self.blocks = nn.Sequential(
            *[
                Block(
                    n_embed,
                    n_head,
                    block_size
                )
                for _ in range(n_layer)
            ]
        )

        self.ln = nn.LayerNorm(
            n_embed
        )

        self.lm_head = nn.Linear(
            n_embed,
            vocab_size
        )


    def forward(
        self,
        idx,
        targets=None
    ):

        B, T = idx.shape

        token_emb = self.token_embedding(
            idx
        )

        x = self.position_encoding(
            token_emb
        )

        x = self.blocks(x)

        x = self.ln(x)

        logits = self.lm_head(x)

        loss = None

        if targets is not None:

            B, T, C = logits.shape

            logits = logits.reshape(
                B * T,
                C
            )

            targets = targets.reshape(
                B * T
            )

            loss = F.cross_entropy(
                logits,
                targets
            )

        return logits, loss