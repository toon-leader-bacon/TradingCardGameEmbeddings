"""Embedding a corpus into an EmbeddingTable, and the view of an encoder
that this needs (CardEmbedder).

CardEmbedder lives here, next to its only consumer. Both encoder models
(src/encoder_model/) satisfy it structurally through CardEncoderModel's
embedding_dim and isolated_embeddings, so there is no adapter class; test
fakes and later embedders (cached, deck-level) just match the Protocol.
"""

import logging
from typing import Iterable, Iterator, Protocol, Sequence
from uuid import UUID

import numpy as np
import torch

from src.encoder_model.precision import Precision
from src.evaluation.card_row import CardRow
from src.evaluation.embedding.embedding_table import (
    EmbeddingTable,
    require_finite_vectors,
)
from src.schema.card import GenericCard

logger = logging.getLogger(__name__)


class CardEmbedder(Protocol):
    """A context-free encoder view: embedding row i depends only on cards[i]."""

    @property
    def embedding_dim(self) -> int:
        """Width of one embedding. Inputs: none. Output: int. Side
        effects: none. Exceptions: none."""
        ...

    def isolated_embeddings(
        self, cards: Sequence[GenericCard], precision: Precision = "fp32"
    ) -> torch.Tensor:
        """Inputs: cards (Sequence[GenericCard], may be empty), precision.
        Output: float32 Tensor (len(cards), embedding_dim) on the CPU.
        Side effects: none lasting (a model runs an eval-mode forward pass
            and restores its mode).
        Exceptions: whatever the underlying encoder raises.
        """
        ...


def embed_corpus(
    embedder: CardEmbedder,
    corpus: Iterable[GenericCard],
    table: EmbeddingTable,
    batch_size: int,
    precision: Precision = "fp32",
    *,
    max_consecutive_failures: int = 5,
) -> int:
    """Embed every corpus card not yet in table, in batches, committing
    each batch.

    Resumable: a card whose nocab_uuid is already stored (or already in
    the current batch) is skipped, so an interrupted run loses at most one
    batch and a second corpus can extend the same table.

    Unattended-safe: a batch whose embedding fails (the embedder raises,
    or returns non-finite values) is logged and skipped - its cards stay
    absent, so a re-run retries them. Only max_consecutive_failures
    failed batches in a row stop the run.

    Inputs: embedder (CardEmbedder), corpus (Iterable[GenericCard], e.g.
        select_corpus(...)), table (an open EmbeddingTable), batch_size
        (>= 1, cards per forward pass), precision for the forward pass,
        max_consecutive_failures (>= 1).
    Output: int, the number of cards added.
    Side effects: writes to table after every successful batch; runs the
        embedder; logs each skipped batch and a final summary.
    Exceptions: ValueError up front if batch_size < 1,
        max_consecutive_failures < 1, or the widths of embedder and table
        differ. Propagates EmbeddingTable.add's ValueError for a card of a
        game the table was not created for (a caller error that would
        recur every batch; earlier batches stay committed).
        RuntimeError after max_consecutive_failures failed batches in a
        row (chained to the last failure). Also propagated, uncaught:
        sqlite3.Error from the table, and whatever iterating corpus raises
        (e.g. a game not loaded in the binder). KeyboardInterrupt
        propagates.

    Example:
        >>> with EmbeddingTable.create(path, metadata) as table:
        ...     embed_corpus(model, select_corpus(binder, spec), table, 64, "fp16")
        12731
    """
    result = 0
    _require_positive("batch_size", batch_size)
    _require_positive("max_consecutive_failures", max_consecutive_failures)
    _require_matching_width(embedder, table)
    consecutive_failures = 0
    skipped_cards = 0

    # Embed and store each batch of new cards; skip a batch that fails
    for batch in _new_card_batches(corpus, table, batch_size):
        try:
            vectors = _finite_embeddings(embedder, batch, precision)
        except Exception as error:
            consecutive_failures += 1
            skipped_cards += len(batch)
            logger.warning("skipped a batch of %d cards", len(batch), exc_info=True)
            if consecutive_failures >= max_consecutive_failures:
                raise RuntimeError(
                    f"{consecutive_failures} consecutive batches failed"
                ) from error
            continue
        consecutive_failures = 0
        table.add(_rows_of(batch), vectors)
        result += len(batch)

    # One summary line, so a partial table is visible in the log
    logger.info("embedded %d cards; skipped %d after failures", result, skipped_cards)
    return result


def _require_positive(name: str, value: int) -> None:
    """Inputs: a parameter's name and value. Output: None. Side effects:
    none. Exceptions: ValueError naming the parameter if value < 1."""
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")


def _require_matching_width(embedder: CardEmbedder, table: EmbeddingTable) -> None:
    """Inputs: embedder, table. Output: None. Side effects: none.
    Exceptions: ValueError naming both widths if embedder.embedding_dim !=
    table.metadata.embedding_dim."""
    table_width = table.metadata.embedding_dim
    if embedder.embedding_dim != table_width:
        raise ValueError(
            f"embedder width {embedder.embedding_dim} != table width {table_width}"
        )


def _new_card_batches(
    corpus: Iterable[GenericCard], table: EmbeddingTable, batch_size: int
) -> Iterator[list[GenericCard]]:
    """Group the corpus cards not yet in table into batches.

    Inputs: corpus, table, batch_size (>= 1).
    Output: Iterator of non-empty lists of at most batch_size distinct
        cards (by nocab_uuid), in corpus order; the last may be shorter.
        Stored cards and repeats within a batch are dropped.
    Side effects: reads table (has_card) as it iterates.
    Exceptions: whatever iterating corpus or table.has_card raises.
    """
    batch: list[GenericCard] = []
    pending: set[UUID] = set()
    for card in corpus:
        if card.nocab_uuid in pending or table.has_card(card.nocab_uuid):
            continue
        batch.append(card)
        pending.add(card.nocab_uuid)
        if len(batch) == batch_size:
            # The caller stores this batch before the next has_card check
            yield batch
            batch, pending = [], set()
    if batch:
        yield batch


def _finite_embeddings(
    embedder: CardEmbedder, batch: Sequence[GenericCard], precision: Precision
) -> np.ndarray:
    """Embed batch and return it as a float32 numpy array.

    Inputs: embedder, batch (non-empty), precision.
    Output: float32 np.ndarray (len(batch), embedding_dim), all finite.
    Side effects: runs the embedder.
    Exceptions: NonFiniteEmbeddingError (via require_finite_vectors) if
        any value is NaN or infinite; whatever the embedder raises.
    """
    # Cast first, exactly as EmbeddingTable.add stores it, so a batch that
    # add() would reject is skipped here instead of ending the run
    with np.errstate(over="ignore"):
        vectors = embedder.isolated_embeddings(batch, precision).numpy()
        vectors = vectors.astype(np.float32, copy=False)
    require_finite_vectors(vectors)
    return vectors


def _rows_of(batch: Sequence[GenericCard]) -> list[CardRow]:
    """Inputs: batch. Output: one CardRow (nocab_uuid, source_game) per
    card, same order. Side effects: none. Exceptions: none."""
    return [CardRow(card.nocab_uuid, card.source_game) for card in batch]
