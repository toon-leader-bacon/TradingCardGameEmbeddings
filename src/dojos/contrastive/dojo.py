"""ContrastiveDojo: the single-card contrastive dojo, this slice's vertical spine.

See plans/contrastive_dojo.md's "The dojo itself and naming" section.
Implements the shared `Dojo` contract (src/dojos/dojo.py): a "batch" is
one ContrastiveBatch built from a sample of decks, an "example" is one
source deck, and the batch budget is translated into a deck count (a
contrastive batch only makes sense whole - its negatives are the other
decks' items - so it is never split by card cost). It has no decoder
head: the loss reads the encoder's embeddings directly.
"""

import logging
from typing import Iterable, Iterator

import torch
from torch import nn

from src.data_refinement.card_binder.card_lookup import CardLookup
from src.data_refinement.card_binder.visible_card_lookup import VisibleCardLookup
from src.dojos.contrastive.contrastive_batch import ContrastiveBatch
from src.dojos.contrastive.contrastive_loss import (
    ContrastiveLoss,
    SingleCardInfoNCELoss,
)
from src.dojos.contrastive.pair_constructor import ContrastivePairConstructor
from src.dojos.dojo import BatchBudget, DojoBatch
from src.dojos.file_managers.DeckBoxDealer import DeckBoxDealer
from src.schema.holdout import HoldoutSpec
from src.schema.splits import Split
from src.schema.type_hints import BatchedModelOutput, iter_cards

_logger = logging.getLogger(__name__)

# InfoNCE needs at least one other deck's items to act as negatives.
_MIN_DECKS_PER_SAMPLE = 2


class ContrastiveDojo:
    """Wires a DeckBoxDealer, a ContrastivePairConstructor, and a
    CardLookup into the `Dojo` contract. Owns its ContrastiveLoss
    internally (an injected collaborator with a sensible default) - same
    convention as every generic dojo cell."""

    def __init__(
        self,
        dealer: DeckBoxDealer,
        pair_constructor: ContrastivePairConstructor,
        card_lookup: CardLookup,
        holdout: HoldoutSpec,
        decks_per_sample: int = 3,
        contrastive_loss: ContrastiveLoss | None = None,
        name: str = "contrastive",
        strict_version_check: bool = True,
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
            card_lookup: the full card store; each split's batches are
                built through a VisibleCardLookup of it.
            holdout: which cards each split may see (must equal the
                training plan's spec).
            decks_per_sample: upper bound on decks per batch; a smaller
                BatchBudget lowers it (see batches()).
            contrastive_loss: the batch-level loss compute_loss
                delegates to. Defaults to SingleCardInfoNCELoss().
            name: this dojo's name in a training plan.
            strict_version_check: when True (default), dealer's
                recorded CardBinder version (dealer.card_binder_version)
                is checked against
                card_lookup.version_for(dealer.source_game) before
                anything else; a mismatch (or a None recorded version)
                raises. False is the explicit escape hatch - it skips
                the check and logs a warning instead.
        Output: none (constructor).
        Side effects: none of its own beyond the check below (one
            logging.warning() when strict_version_check is False).
        Exceptions: ValueError if strict_version_check is True and
            dealer's recorded CardBinder version is missing or doesn't
            match card_lookup's current one.
        """
        self.name = name
        self.holdout = holdout
        self._dealer = dealer
        self._pair_constructor = pair_constructor
        self._decks_per_sample = decks_per_sample
        self._contrastive_loss = contrastive_loss or SingleCardInfoNCELoss()
        self._check_deck_box_version(dealer, card_lookup, strict_version_check)
        self._lookups = {
            split: VisibleCardLookup(card_lookup, holdout, split) for split in Split
        }

    def batches(
        self, split: Split, budget: BatchBudget, max_examples: int | None = None
    ) -> Iterator[ContrastiveBatch]:
        """Yield one split's ContrastiveBatches within `budget`.

        Inputs: split (Split), budget (BatchBudget), max_examples (int |
            None): stop after this many source decks have been dealt.
        Output: iterator of ContrastiveBatch. A degenerate batch (no
            positive clique of size >= 2, e.g. every deck in that sample
            was skipped by the pair constructor for having too few
            visible cards) is logged and skipped rather than yielded.
            Only the TRAIN split reshuffles its decks; TEST/VALIDATION
            deal in fixed order. Item sampling inside the pair
            constructor still advances its own RNG, so a capped pass is
            deterministic in which decks it uses, not in which items.
        Side effects: one logging.warning() per skipped degenerate batch.
        Exceptions: ValueError if the budget can't fit
            _MIN_DECKS_PER_SAMPLE decks, or a built batch's actual cost
            exceeds it; whatever the pair constructor raises.

        Example:
            >>> next(dojo.batches(Split.TRAIN, BatchBudget(64, lambda c: 1)))
        """
        decks_per_sample = self._decks_within(budget)
        for deck_sample in self._dealer.decks_for(
            split, decks_per_sample, shuffle=split == Split.TRAIN
        ):
            if max_examples is not None:
                if max_examples < len(deck_sample):
                    return
                max_examples -= len(deck_sample)
            batch = self._pair_constructor.build(deck_sample, self._lookups[split])
            if not any(len(group) >= 2 for group in batch.positive_cliques):
                _logger.warning(
                    "Skipping a degenerate ContrastiveBatch (no positive "
                    "clique with >= 2 items) built from %d deck(s)",
                    len(deck_sample),
                )
                continue
            self._check_within(batch, budget)
            yield batch

    def example_count(self, split: Split) -> int:
        """Approximate example count: decks in the split (before skips)."""
        return self._dealer.deck_count(split)

    def compute_loss(
        self, item_embeddings: BatchedModelOutput, batch: DojoBatch
    ) -> torch.Tensor:
        """Compute batch's contrastive loss from its items' raw embeddings.

        Inputs:
            item_embeddings: one embedding per batch.inputs entry, same
                order, as returned by the shared encoder's forward()
                call on batch.inputs - shape (single-card vs. multi-card
                item) must match this dojo's contrastive_loss.
            batch: the ContrastiveBatch item_embeddings was computed
                for - supplies identities/positive_cliques to the loss.
        Output: a scalar loss tensor.
        Side effects: none.
        Exceptions: TypeError if batch isn't a ContrastiveBatch; whatever
            self._contrastive_loss.calculate() raises (e.g. ValueError on
            a length or shape mismatch against batch.inputs).

        Example:
            >>> embeddings = encoder.forward(batch.inputs)
            >>> loss = dojo.compute_loss(embeddings, batch)
        """
        if not isinstance(batch, ContrastiveBatch):
            raise TypeError(f"{self.name} needs a ContrastiveBatch")
        return self._contrastive_loss.calculate(
            item_embeddings=item_embeddings,
            identities=batch.identities,
            positive_cliques=batch.positive_cliques,
        )

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        """None: the contrastive loss has no learned head."""
        return []

    def reset_head(self) -> None:
        """No head to reset."""

    def _decks_within(self, budget: BatchBudget) -> int:
        """Decks per sample so a batch fits budget, assuming unit card cost."""
        fit = budget.max_cost // self._pair_constructor.cards_per_deck
        decks = min(self._decks_per_sample, fit)
        if decks < _MIN_DECKS_PER_SAMPLE:
            raise ValueError(
                f"Budget {budget.max_cost} fits only {fit} deck(s) of "
                f"{self._pair_constructor.cards_per_deck} cards; contrastive "
                f"batches need at least {_MIN_DECKS_PER_SAMPLE}"
            )
        return decks

    def _check_within(self, batch: ContrastiveBatch, budget: BatchBudget) -> None:
        """Raise ValueError if the batch's true card cost exceeds budget."""
        cost = sum(budget.cost_of(card) for card in iter_cards(batch.inputs))
        if cost > budget.max_cost:
            raise ValueError(
                f"Contrastive batch costs {cost}, over budget {budget.max_cost}"
            )

    def _check_deck_box_version(
        self, dealer: DeckBoxDealer, card_lookup: CardLookup, strict_version_check: bool
    ) -> None:
        """Raise if dealer's recorded CardBinder version doesn't match
        card_lookup - see __init__'s own docstring for the exact
        contract.

        Private helper - single caller is __init__.
        """
        if not strict_version_check:
            _logger.warning(
                "%s: skipping deck box version check (strict_version_check=False)",
                self.name,
            )
            return

        recorded_version = dealer.card_binder_version
        actual_version = card_lookup.version_for(dealer.source_game)
        if recorded_version != actual_version:
            raise ValueError(
                f"{self.name}: dealer's deck box was minted from CardBinder "
                f"version {recorded_version!r}, but the current one for "
                f"{dealer.source_game} is {actual_version!r} - regenerate "
                "the deck box"
            )
