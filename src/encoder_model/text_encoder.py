"""Turns serialized card text into per-token contextualized embeddings.

TextEncoder is a Strategy injected into SingleCardModel: a pretrained
transformer today, possibly a from-scratch-trained encoder later, without
SingleCardModel itself needing to change. It always returns token-level
output (never pre-pooled) so that swapping the pooling/head architecture
(mean pooling, attention pooling, ...) never requires touching this
contract - see EmbeddingHead in embedding_head.py, which owns pooling.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


@dataclass
class TokenEncoding:
    """One TextEncoder's output for a batch of texts.

    hidden_states: (batch, seq_len, hidden_dim) contextualized per-token
        vectors.
    attention_mask: (batch, seq_len), 1 for a real token, 0 for padding -
        needed by any pooling step so padding doesn't skew it.
    """

    hidden_states: torch.Tensor
    attention_mask: torch.Tensor


class TextEncoder(nn.Module, ABC):
    """Strategy interface: batch of card text in, per-token embeddings out."""

    @abstractmethod
    def encode(self, texts: List[str]) -> TokenEncoding:
        """
        Inputs: texts, a batch of already-serialized card strings (see
            card_serialization.serialize_card_to_json).
        Output: a TokenEncoding with one row of hidden states/mask per
            text, in the same order.
        Side effects: none directly. A trainable (non-frozen) concrete
            encoder's parameters will accumulate gradients as a normal
            side effect of use inside a training loop.
        Exceptions: implementation-defined - typically whatever the
            underlying tokenizer/model raises for malformed input.
        """
        raise NotImplementedError


class PretrainedTextEncoder(TextEncoder):
    """Wraps a pretrained HuggingFace transformers model as a TextEncoder.

    Frozen by default (a fixed feature extractor); set trainable=True to
    fine-tune it end to end alongside the rest of the model instead.
    """

    def __init__(self, checkpoint: str, trainable: bool = False, max_length: int = 512):
        super().__init__()
        self.checkpoint = checkpoint
        self.trainable = trainable
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModel.from_pretrained(checkpoint)

        if not trainable:
            self.model.eval()
            for parameter in self.model.parameters():
                parameter.requires_grad = False

    def encode(self, texts: List[str]) -> TokenEncoding:
        device = next(self.model.parameters()).device
        tokenized = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(device)

        if self.trainable:
            model_out = self.model(**tokenized)
        else:
            with torch.no_grad():
                model_out = self.model(**tokenized)

        return TokenEncoding(
            hidden_states=model_out.last_hidden_state,
            attention_mask=tokenized["attention_mask"],
        )


class StaticEmbeddingTextEncoder(TextEncoder):
    """A from-scratch-trainable TextEncoder: static (context-independent)
    per-token embeddings, with no attention or other cross-token mixing.

    Reuses an existing pretrained tokenizer purely for subword splitting
    (a solved problem not worth re-deriving), while the embedding table
    itself starts randomly initialized and trains from scratch - the
    cheapest possible baseline before adding any real contextualization,
    whether here or via a richer EmbeddingHead.
    """

    def __init__(
        self, tokenizer_checkpoint: str, embedding_dim: int, max_length: int = 512
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.token_embedding = nn.Embedding(
            num_embeddings=self.tokenizer.vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=self.tokenizer.pad_token_id,
        )

    def encode(self, texts: List[str]) -> TokenEncoding:
        device = self.token_embedding.weight.device
        tokenized = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(device)

        return TokenEncoding(
            hidden_states=self.token_embedding(tokenized["input_ids"]),
            attention_mask=tokenized["attention_mask"],
        )
