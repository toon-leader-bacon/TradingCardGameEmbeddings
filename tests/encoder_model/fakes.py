"""A tiny deterministic stand-in for a real language-model text encoder."""

from typing import List

import torch

from src.encoder_model.text_encoder import TextEncoder, TokenEncoding

HIDDEN = 8


class FakeTextEncoder(TextEncoder):
    """Two tokens per text. Each token's features come from the text's
    characters, so equal texts always encode equally, whatever else is in
    the batch."""

    def encode(self, texts: List[str]) -> TokenEncoding:
        rows = []
        for text in texts:
            codes = torch.tensor([float(ord(c)) for c in text])
            token_a = torch.stack(
                [(codes * (i + 1)).sin().mean() for i in range(HIDDEN)]
            )
            token_b = torch.stack(
                [(codes / (i + 2)).cos().mean() for i in range(HIDDEN)]
            )
            rows.append(torch.stack([token_a, token_b]))
        hidden = torch.stack(rows)
        return TokenEncoding(
            hidden_states=hidden, attention_mask=torch.ones(len(texts), 2)
        )
