"""The single-card-in, single-embedding-out model type.

See plans/encoder_model.md for the full narrative design this
codifies. Implemented as a typing.Protocol (structural typing), same
convention as CardIngestionStage/Dojo elsewhere in this project.

A concrete implementation is expected to be a torch.nn.Module (so its
weights can be registered with training/'s optimizer via .parameters()
— mirroring how src/dojos/dojo.py's DecoderHead is typed), which
already satisfies this Protocol structurally as long as it also
defines forward() with this signature. parameters() and __call__() are
declared explicitly below (rather than left implicit) so training/ can
call model.parameters() and model(card) against the SingleCardModel
type itself, without needing to know the concrete implementation is
actually an nn.Module.
"""

from typing import Iterable, Protocol

import torch

from src.encoder_model.embedding import Embedding
from src.schema.card import GenericCard


class SingleCardModel(Protocol):
    """One card in, one embedding out."""

    @property
    def embedding_dim(self) -> int:
        """This model's fixed output embedding width, set at construction time."""
        ...

    def forward(self, card: GenericCard) -> Embedding:
        """Embed a single card.

        Inputs:
            card: the card to embed.
        Output: this card's embedding, shape (embedding_dim,).
        Side effects: none.
        Exceptions: none.
        """
        ...

    def __call__(self, card: GenericCard) -> Embedding:
        """Embed a single card — prefer this over calling forward() directly.

        Inputs:
            card: the card to embed.
        Output: this card's embedding, shape (embedding_dim,).
        Side effects: none.
        Exceptions: none.
        """
        ...

    def parameters(self) -> Iterable[torch.nn.Parameter]:
        """This model's trainable parameters, for the optimizer to register.

        Inputs: none.
        Output: an iterable of this model's trainable parameters.
        Side effects: none.
        Exceptions: none.
        """
        ...
