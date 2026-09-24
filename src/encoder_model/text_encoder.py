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
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

# Post lean_content/compact-JSON serialization, real card text fits well
# under this for every onboarded game (measured: MTG p99 ~296, FaB p99
# ~150 tokens - see src/training/Notes.md section 7); a shorter cap than
# a raw model's own limit both truncates any future noisy source safely
# and keeps a forward pass cheap (attention cost grows with sequence
# length - see src/training/Notes.md section 2).
_DEFAULT_MAX_LENGTH = 384

# Splits an encode() call into same-checkpoint sub-batches of at most this
# many texts, sorted by (character-length) size first, so each sub-batch
# pads to its OWN shorter max length instead of the whole call's longest
# text - measured up to ~4x fewer wasted padding tokens on a mixed batch
# (src/training/Notes.md section 2). Smaller than a typical dojo batch
# (HardwareLimits.max_batch_cost, expected to start ~32) so bucketing
# actually activates; the right value is hardware- and batch-size-
# dependent and worth re-tuning once real training numbers are in.
_DEFAULT_LENGTH_BUCKET_SIZE = 16


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

    A batch longer than length_bucket_size is sorted by text length and
    tokenized/forwarded in length_bucket_size-sized groups rather than one
    padded-to-the-longest-text call - see encode(). This is transparent to
    every caller: the returned TokenEncoding always has one row per input
    text, in the same order, and its attention_mask marks exactly the same
    real/padding tokens a single unbucketed call would have (just possibly
    more trailing padding columns, since every row is still padded out to
    this call's own overall longest text).
    """

    def __init__(
        self,
        checkpoint: str,
        trainable: bool = False,
        max_length: int = _DEFAULT_MAX_LENGTH,
        model_kwargs: dict | None = None,
        length_bucket_size: int | None = _DEFAULT_LENGTH_BUCKET_SIZE,
    ):
        """
        Inputs:
            checkpoint: a Hugging Face model id or local path, used for
                both the tokenizer and the model.
            trainable: False (default) freezes the model and keeps it in
                eval mode, so it acts as a fixed feature extractor.
            max_length: truncation length passed to the tokenizer.
            model_kwargs: extra keyword arguments forwarded to
                AutoModel.from_pretrained (e.g. attn_implementation).
                None (default) passes none.
            length_bucket_size: encode() batches of more than this many
                texts are split into sorted, same-size sub-batches (see
                encode()). None disables this and always tokenizes the
                whole call in one padded batch, matching this class's
                pre-bucketing behavior exactly.
        Output: none (constructor).
        Side effects: downloads/loads checkpoint's tokenizer and model
            weights.
        Exceptions: whatever AutoTokenizer/AutoModel.from_pretrained raise
            for an unknown or malformed checkpoint.
        """
        super().__init__()
        self.checkpoint = checkpoint
        self.trainable = trainable
        self.max_length = max_length
        self.length_bucket_size = length_bucket_size
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModel.from_pretrained(checkpoint, **(model_kwargs or {}))

        if not trainable:
            self.model.eval()
            for parameter in self.model.parameters():
                parameter.requires_grad = False

    def train(self, mode: bool = True) -> "PretrainedTextEncoder":
        """Set train/eval mode, but keep a frozen LM in eval so its dropout
        stays off (it is a fixed feature extractor, not being trained)."""
        super().train(mode)
        if not self.trainable:
            self.model.eval()
        return self

    def encode(self, texts: List[str]) -> TokenEncoding:
        """
        Inputs: texts, a batch of already-serialized card strings.
        Output: see TextEncoder.encode. A batch at or under
            length_bucket_size is tokenized and forwarded in one call, as
            it always was; a larger batch is split into sorted sub-batches
            first (see this class's docstring) - the result is identical
            either way except for how much trailing padding it carries.
        Side effects: none directly (the model may accumulate gradients if
            trainable).
        Exceptions: whatever the underlying tokenizer/model raise.

        Example:
            >>> encoder.encode(["{\\"name\\": \\"Bolt\\"}", "{\\"name\\": \\"Shock\\"}"])
            TokenEncoding(hidden_states=..., attention_mask=...)
        """
        if self.length_bucket_size is None or len(texts) <= self.length_bucket_size:
            return self._encode_batch(texts)
        return self._encode_bucketed(texts)

    def _encode_batch(self, texts: List[str]) -> TokenEncoding:
        """Tokenize and forward the whole given batch in one call, padded
        to its own longest text.

        Private helper - called directly by encode() for a batch at or
        under length_bucket_size, and once per bucket from
        _encode_bucketed() for a larger one.
        """
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

    def _encode_bucketed(self, texts: List[str]) -> TokenEncoding:
        """Encode a batch as several shorter, length-sorted sub-batches.

        Private helper - single consumer is encode(), for a batch longer
        than length_bucket_size. Sorting by raw character length (rather
        than tokenizing every text up front just to measure it) is a
        cheap proxy for token length that is good enough to group similar
        texts together.

        Inputs: texts, a batch of already-serialized card strings, longer
            than length_bucket_size.
        Output: one TokenEncoding for the whole batch, in texts' original
            order - see _merge_bucketed_encodings.
        Side effects: none directly.
        Exceptions: whatever _encode_batch raises.
        """
        bucket_size = self.length_bucket_size
        assert bucket_size is not None  # only called from encode() after that check
        order = sorted(range(len(texts)), key=lambda index: len(texts[index]))
        buckets = [
            order[start : start + bucket_size]
            for start in range(0, len(order), bucket_size)
        ]
        encodings = [
            self._encode_batch([texts[index] for index in bucket]) for bucket in buckets
        ]
        return _merge_bucketed_encodings(encodings, buckets, len(texts))


def _merge_bucketed_encodings(
    encodings: List[TokenEncoding],
    index_groups: List[List[int]],
    batch_size: int,
) -> TokenEncoding:
    """Reassemble per-bucket TokenEncodings into one, in original order.

    Every bucket's own (shorter) sequence length is right-padded with
    zeros up to the longest sequence length across all buckets, so the
    result is one uniform (batch_size, max_seq_len, hidden_dim) tensor -
    the extra padding is marked 0 in attention_mask exactly like ordinary
    tokenizer padding, so it is invisible to any mask-aware pooling.
    Differentiable: index_copy_ has a backward, so gradients reach a
    trainable encoder's parameters through this the same as through a
    single unbucketed forward call.

    Inputs:
        encodings: one TokenEncoding per bucket, in index_groups' order.
        index_groups: encodings[i]'s rows belong at these positions (0
            to batch_size - 1) in the result; every position appears in
            exactly one group.
        batch_size: total rows across every bucket.
    Output: one TokenEncoding, batch_size rows, in original order.
    Side effects: none.
    Exceptions: none expected, given index_groups is a partition of
        range(batch_size).
    """
    max_seq_len = max(encoding.hidden_states.shape[1] for encoding in encodings)
    first = encodings[0]
    hidden_states = first.hidden_states.new_zeros(
        batch_size, max_seq_len, first.hidden_states.shape[2]
    )
    attention_mask = first.attention_mask.new_zeros(batch_size, max_seq_len)
    for encoding, indices in zip(encodings, index_groups):
        pad_amount = max_seq_len - encoding.hidden_states.shape[1]
        index_tensor = torch.tensor(indices, device=hidden_states.device)
        hidden_states.index_copy_(
            0, index_tensor, F.pad(encoding.hidden_states, (0, 0, 0, pad_amount))
        )
        attention_mask.index_copy_(
            0, index_tensor, F.pad(encoding.attention_mask, (0, pad_amount))
        )
    return TokenEncoding(hidden_states=hidden_states, attention_mask=attention_mask)


class StaticEmbeddingTextEncoder(TextEncoder):
    """A from-scratch-trainable TextEncoder: static (context-independent)
    per-token embeddings, with no attention or other cross-token mixing.

    Reuses an existing pretrained tokenizer purely for subword splitting
    (a solved problem not worth re-deriving), while the embedding table
    itself starts randomly initialized and trains from scratch - the
    cheapest possible baseline before adding any real contextualization,
    whether here or via a richer EmbeddingHead. No length_bucket_size here:
    padding costs no attention (there is none), only a few extra embedding
    lookups, so bucketing has nothing worth saving.
    """

    def __init__(
        self,
        tokenizer_checkpoint: str,
        embedding_dim: int,
        max_length: int = _DEFAULT_MAX_LENGTH,
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
