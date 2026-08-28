"""The shared Dojo Strategy contract every training task implements.

See plans/training_pipeline.md for the full narrative design this
codifies. Implemented as a typing.Protocol (structural typing),
matching the convention already established by CardIngestionStage
(src/data_refinement/card_binder/ingestion.py) elsewhere in this
project's architecture, rather than an ABC.

CardCount/CardSet/Split/DojoBatch live here alongside Dojo, not in
training/, because they're part of Dojo's own method signatures — a
dojo implementation needs them regardless of which Trainer eventually
drives it. Card lookup itself (CardLookup) is NOT defined here — it's
card_binder's own read surface minus its write methods, so it's owned
by src/data_refinement/card_binder/card_lookup.py and reused here, not
redefined. A dojo receives it only inside prepare_splits, to translate
whatever raw identifiers its own data uses into GenericCards while
building its private split state — next_batch takes no CardLookup, by
then the dojo already holds everything it needs. This is a deliberate
inversion from an earlier design: a dojo owns its entire data pipeline
end to end (including id translation), rather than training/ doing
generic id resolution on a dojo's behalf — the latter would let one
dojo's ids leak into another dojo's batch, since training/ would have
no way to know whether a given id belongs to that dojo's task.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Generic, Protocol, TypeVar
from uuid import UUID

import torch

from src.data_refinement.card_binder.card_lookup import CardLookup
from src.encoder_model.embedding import Embedding
from src.schema.card import GenericCard

LabelsT = TypeVar("LabelsT")

DecoderHead = torch.nn.Module
# A dojo's own decoder head. training/ only ever calls .parameters(),
# to register it with the optimizer alongside the encoder's own
# weights — it never calls a decoder head's forward pass directly.
# Only the owning dojo's own compute_loss does that.


@dataclass(frozen=True)
class CardCount:
    """How many cards one of a dojo's examples requires.

    A single-card dojo declares CardCount(1, 1) — a strict special
    case of the general contract, not a separate one; every Dojo
    method operates on lists regardless (see DojoBatch/compute_loss
    below), so a single-card dojo runs unmodified under a multi-card
    Trainer.

    Inputs: none (data holder).
    Output: n/a.
    Side effects: none.
    Exceptions: none.
    """

    minimum: int
    maximum: int | None  # None = unbounded


@dataclass(frozen=True)
class CardSet:
    """One dojo example's input cards, in dojo-defined order.

    len(cards) must fall within the owning dojo's CardCount.

    Inputs: none (data holder).
    Output: n/a.
    Side effects: none.
    Exceptions: none.
    """

    cards: list[GenericCard]


class Split(Enum):
    """Which of a dojo's three example pools a batch is drawn from.

    TRAIN: the only split a Trainer draws from with an eye toward
        calling .backward() afterward. next_batch(TRAIN, ...) samples
        indefinitely (with replacement across calls) — no
        exhaustiveness guarantee, since a Trainer draws as many
        batches as its own step count calls for, independent of split
        size.
    TEST: a diagnostic-only pass, run once after training finishes
        (not interleaved with TRAIN steps, not currently acted on —
        see plans/training_pipeline.md's Open Questions for the
        deferred adaptive-use case). next_batch(TEST, ...) MUST behave
        like the VALIDATE case below.
    VALIDATE: touched exactly once, after TEST, for this run's
        reported result — never touched during TRAIN or TEST.
        next_batch(VALIDATE, ...) MUST be a stateful cursor that does
        not repeat or skip an example within one full pass — exactly
        ceil(split_size(VALIDATE) / batch_size) consecutive calls —
        and wraps only once that full pass completes. The final call
        in a pass returns a SHORT batch (truncated, not padded) when
        split_size(split) % batch_size != 0.

    Inputs: none (enum).
    Output: n/a.
    Side effects: none.
    Exceptions: none.
    """

    TRAIN = "train"
    TEST = "test"
    VALIDATE = "validate"


@dataclass(frozen=True)
class DojoBatch(Generic[LabelsT]):
    """One training step's worth of already-assembled input + ground truth.

    card_sets and labels are the same length and same order —
    labels[i] is card_sets[i]'s ground truth.

    Inputs: none (data holder).
    Output: n/a.
    Side effects: none.
    Exceptions: none.
    """

    card_sets: list[CardSet]
    labels: list[LabelsT]


class Dojo(Protocol[LabelsT]):
    """One pluggable auxiliary training task.

    LabelsT is fixed concretely by each implementation (e.g.
    GameClassificationDojo fixes it to GameId) — training/ never
    inspects LabelsT's shape, it only round-trips whatever a
    DojoBatch.labels carried back into that same dojo's compute_loss.
    A dojo's decoder head, loss computation, and label shape are all
    private/opaque to the dojo itself; nothing outside this dojo's own
    directory should ever need to know any of them.
    """

    def card_count(self) -> CardCount:
        """How many cards one of this dojo's examples requires.

        Inputs: none.
        Output: this dojo's CardCount.
        Side effects: none.
        Exceptions: none.
        """
        ...

    def prepare_splits(self, corpus: CardLookup, held_out_cards: set[UUID]) -> None:
        """Build and privately cache this dojo's train/test/validate example pools.

        held_out_cards is a project-level, curated set of nocab_uuids
        (see plans/training_pipeline.md's Axis B) an implementation
        may optionally honor by excluding any example whose CardSet
        would touch one of them from the train pool. Ignoring
        held_out_cards entirely is a valid implementation choice for a
        dojo that isn't testing novel-card generalization.

        Inputs:
            corpus: card lookup to translate whatever raw identifiers
                this dojo's own data uses into GenericCards, while
                building its internal split state.
            held_out_cards: nocab_uuids to keep out of the train pool,
                if this dojo chooses to honor Axis B.
        Output: none — mutates this dojo's own internal state, later
            read by split_size/next_batch.
        Side effects: none beyond this dojo's own internal state.
        Exceptions: none.
        """
        ...

    def split_size(self, split: Split) -> int:
        """How many examples are in one of this dojo's example pools.

        Inputs:
            split: which pool to size.
        Output: that pool's example count, as of the last
            prepare_splits call.
        Side effects: none.
        Exceptions: raises RuntimeError if called before
            prepare_splits.
        """
        ...

    def next_batch(self, split: Split, batch_size: int) -> DojoBatch[LabelsT]:
        """Produce one already-assembled, already-labeled batch from split.

        See Split's own docstring for TRAIN vs. TEST/VALIDATE
        behavior — TRAIN samples indefinitely; TEST/VALIDATE must
        cover their pool exactly once per full pass, with a short
        final batch rather than padding.

        Inputs:
            split: which pool to draw from.
            batch_size: how many examples to draw.
        Output: a DojoBatch whose card_sets/labels are the same
            length (batch_size, except a TEST/VALIDATE pass's final
            call) and order.
        Side effects: advances this dojo's internal cursor for split.
        Exceptions: raises RuntimeError if called before
            prepare_splits.
        """
        ...

    def build_decoder_head(self, embedding_dim: int) -> DecoderHead:
        """Construct this dojo's own decoder head for a given embedding size.

        Inputs:
            embedding_dim: the encoder's output embedding width.
        Output: a fresh, trainable DecoderHead sized for embedding_dim.
        Side effects: none.
        Exceptions: none.
        """
        ...

    def compute_loss(
        self, embeddings: list[list[Embedding]], labels: list[LabelsT]
    ) -> torch.Tensor:
        """Compute this batch's loss from encoder output and ground truth.

        Runs this dojo's own decoder head (and, internally, whatever
        Loss object this dojo was constructed with) — both stay
        entirely private to this call; training/ never invokes either
        directly.

        Inputs:
            embeddings: one list[Embedding] per example, in the same
                order and CardSet-length as the DojoBatch this batch's
                labels came from — embeddings[i][j] is the encoder's
                output for that example's j-th card.
            labels: this batch's labels, as returned by next_batch.
        Output: a scalar loss tensor for this batch.
        Side effects: none — does not call .backward() or step any
            optimizer; that's training/'s job.
        Exceptions: none.
        """
        ...
