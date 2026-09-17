"""ContrastiveDojo: the single-card contrastive dojo, this slice's vertical spine.

See plans/contrastive_dojo.md's "The dojo itself and naming" section.
Placeholder name - a genuine third task shape ("a batch, jointly, in;
one loss out"), distinct from the existing Dojo's "one example, one
label, one loss." Presents training_data()/test_data()/validation_data()
generators yielding ContrastiveBatch (mirroring the existing Dojo
generator convention), but its loss entry point takes the batch-level
shape below instead of Dojo's compute_loss(embeddings, labels) - a
deliberate contract break, not an oversight.
"""

import logging
from typing import Iterator

import torch

from src.data_refinement.card_binder.card_lookup import CardLookup
from src.dojos.contrastive.contrastive_batch import ContrastiveBatch
from src.dojos.contrastive.contrastive_loss import (
    ContrastiveLoss,
    SingleCardInfoNCELoss,
)
from src.dojos.contrastive.deck_box_dealer import DeckBoxDealer
from src.dojos.contrastive.pair_constructor import ContrastivePairConstructor
from src.schema.type_hints import BatchedModelOutput

_logger = logging.getLogger(__name__)


class ContrastiveDojo:
    """Wires a DeckBoxDealer, a ContrastivePairConstructor, and a
    CardLookup into the train/test/validation split + loss-computation
    contract a training loop drives. Owns its ContrastiveLoss internally
    (an injected collaborator with a sensible default, never constructed
    ad hoc by callers) - same convention as every existing generic dojo
    cell."""

    def __init__(
        self,
        dealer: DeckBoxDealer,
        pair_constructor: ContrastivePairConstructor,
        card_lookup: CardLookup,
        decks_per_sample: int = 3,
        contrastive_loss: ContrastiveLoss | None = None,
    ) -> None:
        """
        Inputs:
            dealer: source of raw deck samples for every split.
            pair_constructor: turns one deck sample into one
                ContrastiveBatch - the swappable research surface (see
                pair_constructor.py). Its item shape must match
                contrastive_loss's expected item_embeddings shape - a
                dojo wiring invariant contrastive_loss.calculate()
                enforces at call time, not this constructor.
            card_lookup: passed through to pair_constructor.build() on
                every deck sample.
            decks_per_sample: how many decks dealer deals per sample,
                for every split (this slice's default: 3).
            contrastive_loss: the batch-level loss compute_loss
                delegates to. Defaults to SingleCardInfoNCELoss().
        Output: none (constructor).
        Side effects: none.
        Exceptions: none.
        """
        self._dealer = dealer
        self._pair_constructor = pair_constructor
        self._card_lookup = card_lookup
        self._decks_per_sample = decks_per_sample
        self._contrastive_loss = contrastive_loss or SingleCardInfoNCELoss()

    def training_data(self) -> Iterator[ContrastiveBatch]:
        """Yield the training split's data as ContrastiveBatches.

        Inputs: none.
        Output: a generator of ContrastiveBatch, one per deck sample
            dealt by self._dealer.training_decks() - see _batches_from()'s
            docstring for the one case that's skipped rather than
            yielded.
        Side effects: emits one logging.warning() per skipped
            degenerate batch, beyond self._dealer/self._pair_constructor's own.
        Exceptions: whatever self._pair_constructor.build() raises.
        """
        yield from self._batches_from(
            self._dealer.training_decks(self._decks_per_sample)
        )

    def test_data(self) -> Iterator[ContrastiveBatch]:
        """Yield the test split's data as ContrastiveBatches.

        Inputs: none.
        Output: a generator of ContrastiveBatch, one per deck sample
            dealt by self._dealer.test_decks() - see _batches_from()'s
            docstring for the one case that's skipped rather than
            yielded.
        Side effects: emits one logging.warning() per skipped
            degenerate batch, beyond self._dealer/self._pair_constructor's own.
        Exceptions: whatever self._pair_constructor.build() raises.
        """
        yield from self._batches_from(self._dealer.test_decks(self._decks_per_sample))

    def validation_data(self) -> Iterator[ContrastiveBatch]:
        """Yield the validation split's data as ContrastiveBatches.

        Inputs: none.
        Output: a generator of ContrastiveBatch, one per deck sample
            dealt by self._dealer.validation_decks() - see _batches_from()'s
            docstring for the one case that's skipped rather than
            yielded.
        Side effects: emits one logging.warning() per skipped
            degenerate batch, beyond self._dealer/self._pair_constructor's own.
        Exceptions: whatever self._pair_constructor.build() raises.
        """
        yield from self._batches_from(
            self._dealer.validation_decks(self._decks_per_sample)
        )

    def compute_loss(
        self, item_embeddings: BatchedModelOutput, batch: ContrastiveBatch
    ) -> torch.Tensor:
        """Compute batch's contrastive loss from its items' raw embeddings.

        Inputs:
            item_embeddings: one embedding per batch.items entry, same
                order, as returned by the shared encoder's forward()
                call on batch.items - shape (single-card vs. multi-card
                item) must match this dojo's contrastive_loss.
            batch: the ContrastiveBatch item_embeddings was computed
                for - supplies identities/positive_cliques to the loss.
        Output: a scalar loss tensor.
        Side effects: none.
        Exceptions: whatever self._contrastive_loss.calculate() raises
            (e.g. ValueError on a length or shape mismatch against
            batch.items).

        Example:
            >>> embeddings = encoder.forward(batch.items)
            >>> loss = dojo.compute_loss(embeddings, batch)
        """
        return self._contrastive_loss.calculate(
            item_embeddings=item_embeddings,
            identities=batch.identities,
            positive_cliques=batch.positive_cliques,
        )

    def _batches_from(self, deck_samples: Iterator[list]) -> Iterator[ContrastiveBatch]:
        """Shared deck-sample-to-ContrastiveBatch conversion for
        training_data/test_data/validation_data.

        Private helper - single set of callers are the three public
        split methods above. Mirrors the existing generic dojo cells'
        `_data_iterator` convention.

        Inputs:
            deck_samples: a generator of deck samples, one per
                self._decks_per_sample decks, from self._dealer.
        Output: a generator of ContrastiveBatch, one per deck sample -
            except a degenerate batch (no positive_cliques entry of size
            >= 2, e.g. every deck in that sample was itself skipped by
            the pair constructor for having too few known cards), which
            is logged and skipped rather than yielded. Left un-skipped,
            such a batch would only fail much later, deep inside
            ContrastiveLoss.calculate() - too late to recover from
            mid-training without losing that step's whole batch anyway.
        Side effects: emits one logging.warning() per skipped degenerate
            batch, beyond self._pair_constructor.build()'s own.
        Exceptions: whatever self._pair_constructor.build() raises.
        """
        for deck_sample in deck_samples:
            batch = self._pair_constructor.build(deck_sample, self._card_lookup)
            if not any(len(group) >= 2 for group in batch.positive_cliques):
                _logger.warning(
                    "Skipping a degenerate ContrastiveBatch (no positive "
                    "clique with >= 2 items) built from %d deck(s)",
                    len(deck_sample),
                )
                continue
            yield batch
