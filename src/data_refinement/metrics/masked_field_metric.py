"""Template Method base for masking metrics: one output row per
(eligible card, masked field) - see _MaskedFieldRow below.

See plans/masking_metrics.md for the full design. A subclass fixes
which field is masked (MASKED_FIELD), what its fixed label vocabulary
is (LABEL_VALUES), and two decisions:

  - _is_eligible(): whether a card is a valid sample for this field at
    all (THE ELIGIBILITY DECISION LIVES HERE - see the plan's
    "Eligibility filtering is real logic, not ceremony" section. A
    Dojo reading this metric's output may still skip a row
    defensively, but that's a backstop, never the primary mechanism).
  - _label_for_card(): the true label for an already-eligible card.

scan() owns the shared sequence (iterate the corpus, filter, look up
labels, write parquet) - that part is not a subclass's job to repeat.

NOT A Metric[RawRowT] (metric.py): satisfies CorpusScanMetric
(corpus_scan_metric.py) instead. See that module's docstring for why -
in short, a CardBinder is already fully loaded by scan() time, so
there's no external per-row source to accumulate() over.
"""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import ClassVar

import pandas as pd

from src.data_refinement.card_binder.card_lookup import CardLookup
from src.schema.card import GenericCard
from src.schema.game_id import GameId

OTHER_LABEL = "OTHER"  # shared catch-all class for a numeric-field
# subclass's LABEL_VALUES - see MaskedFieldMetric._label_or_other().


@dataclass(frozen=True)
class _MaskedFieldRow:
    """One typed output row: an eligible card, the field masked out of
    it, and its true label - the same three values scan() writes into
    one parquet row.

    Inputs: none (data holder).
    Output: n/a.
    Side effects: none.
    Exceptions: none.
    """

    nocab_uuid: str
    masked_field: list[str]
    label: str


class MaskedFieldMetric(ABC):
    """Per-card masked-field classification target: one row per
    eligible card, giving the true value of a field a Dojo will later
    mask out of that card's raw_content.

    Satisfies the CorpusScanMetric Protocol (corpus_scan_metric.py)
    structurally.
    """

    SOURCE_GAME: ClassVar[GameId]
    MASKED_FIELD: ClassVar[list[str]]
    LABEL_VALUES: ClassVar[tuple[str, ...]]
    DEFAULT_OUTPUT_PATH: ClassVar[Path]

    def __init__(
        self,
        card_lookup: CardLookup,
        output_path: Path | None = None,
    ) -> None:
        """
        Inputs:
            card_lookup: already-populated registry to scan - must
                already have self.SOURCE_GAME's cards ingested. Typed
                as CardLookup (card_binder/card_lookup.py), not the
                full CardBinder, since this class only ever calls
                all_cards() - see plans/masking_metrics.md's
                constructor rationale (a real CardBinder satisfies
                CardLookup structurally, so callers pass one
                directly - no adapter needed).
            output_path: overrides DEFAULT_OUTPUT_PATH when given.
        Output: none (constructor).
        Side effects: none - no I/O happens until scan() is called.
        Exceptions: none.
        """
        self._card_lookup = card_lookup
        self._output_path = output_path or self.DEFAULT_OUTPUT_PATH

    def scan(self) -> Path:
        """Walk every self.SOURCE_GAME card in self._card_lookup, keep
        the ones _is_eligible() accepts, and write one row per
        eligible card to self._output_path.

        Inputs: none (uses self._card_lookup, supplied at
            construction).
        Output: self._output_path.
        Side effects: creates self._output_path's parent directories
            if missing; writes self._output_path (a parquet file with
            columns nocab_uuid: str, masked_field: list[str] - a
            native pyarrow list<string> column, inferred automatically
            from a python list value per cell, not a JSON-encoded
            string - and label: str). One row per eligible card; no
            sample_count column, since this isn't an aggregate.
        Exceptions: whatever pandas.DataFrame.to_parquet raises.

        Example:
            >>> metric = FactionMaskMetric(card_lookup)
            >>> metric.scan()
            PosixPath('data/metrics/gwent_one/faction_mask.parquet')
        """
        result: list[_MaskedFieldRow] = []

        # Walk every card of this metric's game, keeping only the
        # ones this subclass considers a valid sample for its field.
        for card in self._card_lookup.all_cards(self.SOURCE_GAME):
            if not self._is_eligible(card):
                continue
            result.append(self._mask_row(card))

        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([asdict(row) for row in result]).to_parquet(
            self._output_path, index=False
        )
        return self._output_path

    def _mask_row(self, card: GenericCard) -> _MaskedFieldRow:
        """Build one output row for a single already-eligible card.

        Private helper - single consumer is scan().

        Inputs:
            card: a card _is_eligible() has already accepted.
        Output: a _MaskedFieldRow - nocab_uuid (str(card.nocab_uuid)),
            masked_field (a copy of self.MASKED_FIELD), and label
            (self._label_for_card(card)).
        Side effects: none.
        Exceptions: whatever _label_for_card() raises.
        """
        return _MaskedFieldRow(
            nocab_uuid=str(card.nocab_uuid),
            masked_field=list(self.MASKED_FIELD),
            label=self._label_for_card(card),
        )

    def _raw_field_value(self, card: GenericCard) -> str:
        """Walk self.MASKED_FIELD as a path into card.raw_content and
        return the value found there.

        Private helper - shared by every subclass's _label_for_card(),
        so MASKED_FIELD is declared once and walked once here, rather
        than each subclass separately hardcoding the same key as a
        literal string (which would let MASKED_FIELD and the actual
        lookup silently drift apart - see plans/masking_metrics.md:
        MASKED_FIELD is meant to be a walkable path, not just
        documentation).

        Inputs:
            card: a card to walk raw_content on.
        Output: the raw string value found at self.MASKED_FIELD's
            path (e.g. card.raw_content["faction"] for ["faction"];
            card.raw_content["stat_block"]["attack"] for a future
            ["stat_block", "attack"]).
        Side effects: none.
        Exceptions: raises KeyError if any path segment is missing
            from card.raw_content.
        """
        node: dict = card.raw_content
        for key in self.MASKED_FIELD[:-1]:
            node = node[key]
        return node[self.MASKED_FIELD[-1]]

    def _label_or_other(self, raw_value: str) -> str:
        """Encode raw_value as a classification label, falling back to
        OTHER_LABEL for anything outside self.LABEL_VALUES.

        Private helper - shared by every numeric-field subclass's
        _label_for_card() (e.g. ProvisionMaskMetric, PowerMaskMetric,
        ArmorMaskMetric), so the "unknown value -> OTHER" rule is
        written once here rather than repeated identically per
        subclass.

        Inputs:
            raw_value: a candidate label value, typically
                self._raw_field_value(card).
        Output: raw_value itself if it's a member of self.LABEL_VALUES,
            otherwise OTHER_LABEL.
        Side effects: none.
        Exceptions: none.
        """
        if raw_value not in self.LABEL_VALUES:
            return OTHER_LABEL
        return raw_value

    def _is_eligible(self, card: GenericCard) -> bool:
        """Whether card is a valid sample for this metric's masked
        field.

        Default: every card is eligible. Override when a field
        doesn't apply to every card (e.g. PowerMaskMetric/
        ArmorMaskMetric restricting to unit-type cards,
        ProvisionMaskMetric excluding stratagem cards) - see
        plans/masking_metrics.md's "Eligibility filtering is real
        logic, not ceremony" section for why this decision belongs
        here rather than left to a Dojo to discover defensively.

        Inputs:
            card: one card from
                self._card_lookup.all_cards(self.SOURCE_GAME).
        Output: True if card should get an output row.
        Side effects: implementation-defined (expected: none - a
            plain raw_content field check).
        Exceptions: implementation-defined.
        """
        return True

    @abstractmethod
    def _label_for_card(self, card: GenericCard) -> str:
        """The true label for an already-eligible card.

        Only ever called (via _mask_row()) after _is_eligible(card)
        has returned True. Expected to be implemented via
        self._raw_field_value(card), encoded/validated against
        self.LABEL_VALUES as this subclass's field requires.

        Inputs:
            card: one eligible card.
        Output: a value that IS a member of self.LABEL_VALUES (a
            numeric-field subclass returns "OTHER" itself for a value
            outside its known encoding, rather than raising).
        Side effects: implementation-defined (expected: none).
        Exceptions: implementation-defined.
        """
        raise NotImplementedError
