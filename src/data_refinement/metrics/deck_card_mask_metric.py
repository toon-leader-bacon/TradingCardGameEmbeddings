"""Template Method base for masking metrics whose training input is a
whole card removed from a deck, chosen via game-specific logic - not a
field of one card (see masked_field_metric.py for that sibling
pattern) and not a random card (the "which card" decision is real
domain logic a subclass owns, e.g. "the leader").

See plans/deck_card_masking.md for the full design, in particular why
this is raw-row-driven (Metric[dict]-shaped, mirroring
sts_gg/deck_label_metric.py::DeckLabelMetric) rather than driven by
iterating an already-populated DeckBox: GenericDeck/DeckBox are
deliberately game-agnostic (a plain card_nocab_uuids multiset, no
role/slot information), so a deck already sitting in a DeckBox has
lost whatever game-specific structure (e.g. "which entry was the
leader") its raw source made explicit. Every subclass therefore reads
its target directly off the RAW ROW (e.g. a play_gwent guide's own
"leaderId" field), not by re-deriving it from a deck's card list.

A subclass fixes:

  - _deck_uuid_for_row(): ensures this row's deck exists in deck_box
    (a side effect - e.g. delegating to the game's own
    DeckExtractionStage, to avoid duplicating that class's card-
    resolution/uuid-minting logic) and returns its uuid. Every row
    accumulate() sees gets this call, whether or not it also has a
    valid masking target - the deck box is kept complete regardless.
  - _target_card_uuid_for_row(): which single card this metric masks
    out for this row - game-specific and raw-row-aware (Strategy).
    None means this row has no valid target and gets no output row -
    a real, expected outcome, not an error.
  - _label_for_card(): the true label for that target card, once
    picked. Label semantics are DELIBERATELY per-metric (see the
    plan's "Label semantics" section) - unlike MaskedFieldMetric,
    there is no shared MASKED_FIELD-path convention here.

accumulate() owns the shared per-row sequence; finalize() just closes
the writer. Mirrors DeckLabelMetric's streaming shape exactly - see
that class's own docstring for why this satisfies Metric[dict]
(metric.py) structurally rather than CorpusScanMetric.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar
from uuid import UUID

import pyarrow as pa
import pyarrow.parquet as pq

from src.data_refinement.card_binder.card_lookup import CardLookup
from src.data_refinement.deck_box.deck_box import DeckBox
from src.schema.card import GenericCard
from src.schema.game_id import GameId


class DeckCardMaskMetric(ABC):
    """One raw row -> (deck_uuid, target_card_uuid, label), one row
    per raw record with a valid masking target, written as soon as
    accumulate() sees it. Also ensures every row's deck exists in
    deck_box, regardless of whether that row has a valid target.

    Structural Metric[dict] (metric.py) - every concrete subclass
    satisfies it, since accumulate()/finalize() are defined here and
    inherited unchanged.
    """

    SOURCE_GAME: ClassVar[GameId]
    LABEL_VALUES: ClassVar[tuple[str, ...]]
    DEFAULT_OUTPUT_PATH: ClassVar[Path]

    def __init__(
        self,
        card_lookup: CardLookup,
        deck_box: DeckBox,
        output_path: Path | None = None,
    ) -> None:
        """
        Inputs:
            card_lookup: already-populated registry used to inspect a
                target card's raw_content - must already have
                self.SOURCE_GAME's cards ingested. Never written to.
            deck_box: the deck store every row's deck is written into
                as accumulate() sees it (see _deck_uuid_for_row()) -
                may be empty at construction, or already the published
                data/final/decks/<game>.jsonl box loaded via
                DeckBox.load() (writes are idempotent either way, by
                construction of the game's own deck-identity scheme).
            output_path: overrides DEFAULT_OUTPUT_PATH when given.
        Output: none (constructor).
        Side effects: creates output_path's parent directories if
            missing; opens output_path for writing (truncating any
            existing file) via a pyarrow.parquet.ParquetWriter held
            open for the lifetime of this instance - callers MUST call
            finalize() when done, or the file is left incomplete.
        Exceptions: whatever pyarrow.parquet.ParquetWriter raises on
            failure to open output_path for writing.
        """
        self._card_lookup = card_lookup
        self._deck_box = deck_box
        self._output_path = output_path or self.DEFAULT_OUTPUT_PATH
        self._output_schema = pa.schema(
            [
                ("deck_uuid", pa.string()),
                ("target_card_uuid", pa.string()),
                ("label", pa.string()),
            ]
        )
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = pq.ParquetWriter(self._output_path, self._output_schema)
        self._closed = False

    def accumulate(self, row: dict) -> None:
        """Ensure row's deck exists in self._deck_box, then write one
        output row if row has a valid masking target.

        Inputs:
            row: one raw record from this metric's own game-specific
                raw source (e.g. one parsed JSON guide object from
                play_gwent's guides.jsonl).
        Output: none.
        Side effects: whatever self._deck_uuid_for_row() does (at
            minimum, ensures row's deck is stored in self._deck_box).
            Writes exactly one row to the open ParquetWriter when
            self._target_card_uuid_for_row() finds a target; writes
            nothing when it returns None.
        Exceptions: whatever _deck_uuid_for_row()/
            _target_card_uuid_for_row()/_label_for_card() raise.
            Raises RuntimeError if _target_card_uuid_for_row() returns
            a uuid self._card_lookup cannot find a card for - an
            internal contract violation, distinct from returning None
            itself (which is a real, expected "no target" outcome).

        Example:
            >>> metric = LeaderMaskedFromDeckMetric(card_lookup, deck_box)
            >>> metric.accumulate(json.loads(next(guides_file)))
            >>> metric.finalize()
        """
        # Every row's deck is kept up to date in the box regardless of
        # whether this row also has a masking target.
        deck_uuid = self._deck_uuid_for_row(row, self._deck_box)

        target_uuid = self._target_card_uuid_for_row(row, self._card_lookup)
        if target_uuid is None:
            # No valid target for this row - a real, expected outcome
            # (see _target_card_uuid_for_row()'s contract), not every
            # row produces a training example.
            return

        card = self._card_lookup.get_by_uuid(target_uuid)
        if card is None:
            raise RuntimeError(
                f"{type(self).__name__}: _target_card_uuid_for_row() returned "
                f"{target_uuid}, which card_lookup could not find"
            )
        label = self._label_for_card(card)

        output_row = pa.Table.from_pydict(
            {
                "deck_uuid": [str(deck_uuid)],
                "target_card_uuid": [str(target_uuid)],
                "label": [label],
            },
            schema=self._output_schema,
        )
        self._writer.write_table(output_row)

    def finalize(self) -> Path:
        """Close the underlying ParquetWriter.

        Idempotent: a second call is a no-op. Does NOT save
        self._deck_box - that's the calling driver's own
        responsibility (mirrors DeckLabelMetric.finalize()).

        Inputs: none.
        Output: self._output_path.
        Side effects: closes the ParquetWriter opened in __init__, if
            not already closed.
        Exceptions: whatever ParquetWriter.close() raises.

        Example:
            >>> metric.finalize()
            PosixPath('data/metrics/play_gwent/leader_masked_from_deck.parquet')
        """
        if not self._closed:
            self._writer.close()
            self._closed = True
        return self._output_path

    @abstractmethod
    def _deck_uuid_for_row(self, row: dict, deck_box: DeckBox) -> UUID:
        """Ensure row's deck exists in deck_box, and return its uuid.

        Called once per row, unconditionally - every row's deck is
        kept up to date in deck_box regardless of whether this row
        also has a valid masking target. Expected to delegate to the
        game's own DeckExtractionStage (e.g.
        PlayGwentDeckExtractionStage.extract_one() for the deck-box
        side effect, plus its deck_uuid_for_guide() for the uuid
        itself, since extract_one() only returns a uuid when it
        changed something) rather than re-implementing card
        resolution/uuid-minting here.

        Inputs:
            row: one raw record, same row accumulate() received.
            deck_box: same box passed to __init__.
        Output: the uuid of row's deck, now guaranteed present in
            deck_box.
        Side effects: implementation-defined (expected: creates or
            updates exactly one deck on deck_box).
        Exceptions: implementation-defined (expected: raises if row is
            missing a field this subclass's deck extraction needs).
        """
        raise NotImplementedError

    @abstractmethod
    def _target_card_uuid_for_row(
        self, row: dict, card_lookup: CardLookup
    ) -> UUID | None:
        """Pick the single card this metric masks out for row.

        The game-specific "which card" decision (Strategy), read
        directly off row's own raw fields (e.g. a play_gwent guide's
        "leaderId") - never re-derived from a deck already sitting in
        a DeckBox (see this module's docstring for why).

        Inputs:
            row: one raw record, same row accumulate() received.
            card_lookup: same registry passed to __init__.
        Output: the target card's nocab_uuid, or None if row has no
            valid target - a real, expected outcome that causes
            accumulate() to skip writing an output row for it, not an
            error condition.
        Side effects: implementation-defined (expected: none - reads
            only).
        Exceptions: implementation-defined.
        """
        raise NotImplementedError

    @abstractmethod
    def _label_for_card(self, card: GenericCard) -> str:
        """The true label for an already-picked target card.

        Only ever called (via accumulate()) after
        _target_card_uuid_for_row() has returned a non-None uuid and
        card_lookup has looked up the matching GenericCard for it.
        Label semantics are deliberately per-metric - see
        plans/deck_card_masking.md's "Label semantics" section for why
        this class does not standardize a MASKED_FIELD-path convention
        the way MaskedFieldMetric does.

        Inputs:
            card: the target card _target_card_uuid_for_row() picked.
        Output: a value expected to be a member of self.LABEL_VALUES -
            how an out-of-vocabulary value is handled (e.g. an
            OTHER-style fallback) is this subclass's own decision.
        Side effects: implementation-defined (expected: none).
        Exceptions: implementation-defined.
        """
        raise NotImplementedError
