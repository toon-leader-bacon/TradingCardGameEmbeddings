"""A minimal SingleCardModel implementation, for exercising the training slice.

Not a serious architecture — just enough of a torch.nn.Module to prove
out the SingleCardModel/Dojo/Trainer contract end to end. A real
architecture (text serialization of raw_content through a language
model, etc.) is a separate, later design.
"""

import hashlib

import torch

from src.schema.card import GenericCard


class ToySingleCardModel(torch.nn.Module):
    """Embeds a card by hashing its name into a fixed-size lookup table.

    Satisfies SingleCardModel (src/encoder_model/single_card/
    single_card_model.py) structurally via forward()/__call__()/
    parameters(), all inherited from torch.nn.Module except forward().
    """

    def __init__(self, embedding_dim: int, vocab_size: int) -> None:
        """Construct a hashed-embedding-table toy model.

        Inputs:
            embedding_dim: output embedding width.
            vocab_size: size of the internal hashed lookup table.
        Output: none (constructor).
        Side effects: registers this module's own nn.Embedding table
            as a trainable parameter.
        Exceptions: none.
        """
        super().__init__()
        self._embedding_dim = embedding_dim
        self._vocab_size = vocab_size
        self._table = torch.nn.Embedding(vocab_size, embedding_dim)

    @property
    def embedding_dim(self) -> int:
        """See SingleCardModel.embedding_dim."""
        return self._embedding_dim

    def forward(self, card: GenericCard) -> torch.Tensor:
        """Embed card by hashing its name into this model's lookup table.

        Inputs:
            card: the card to embed.
        Output: this card's embedding, shape (embedding_dim,).
        Side effects: none.
        Exceptions: none.

        Example:
            >>> model = ToySingleCardModel(embedding_dim=8, vocab_size=64)
            >>> model(some_card).shape
            torch.Size([8])
        """
        index = self._hash_card_name(card.name)
        return self._table(torch.tensor(index, dtype=torch.long))

    def _hash_card_name(self, name: str) -> int:
        """Map a card name to a stable index into this model's lookup table.

        Private helper — single consumer is forward(). Uses hashlib
        rather than Python's built-in hash(), which is randomized per
        process (PYTHONHASHSEED) for str and would make the same card
        name map to a different index across runs.

        Inputs:
            name: the card name to hash.
        Output: an index in [0, vocab_size).
        Side effects: none.
        Exceptions: none.
        """
        digest = hashlib.sha256(name.encode("utf-8")).hexdigest()
        return int(digest, 16) % self._vocab_size
