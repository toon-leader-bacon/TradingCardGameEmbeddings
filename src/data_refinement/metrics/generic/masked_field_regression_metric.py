"""Template Method base for regression-style masking metrics: one
output row per (eligible card, masked field), with a float label
instead of a fixed-set classification label.

Mirrors MaskedFieldMetric (masked_field_metric.py) - same scan()
sequence, same _is_eligible()/eligibility-lives-here philosophy - but
deliberately a separate class rather than one generic base
parameterized over label dtype: LABEL_VALUES/_label_or_other's
OTHER-bucket fallback is a real, classification-only concept with no
regression equivalent. A regression subclass's only tool for excluding
a card whose field value can't be treated as a number is _is_eligible()
itself (e.g. dominiontabs/cost_regression_metric.py excluding a
variable/minimum/uncosted card's non-digit cost string) - there's no
OTHER bucket to fall back into.

NOT A Metric[RawRowT] (../metric.py): satisfies CorpusScanMetric
(corpus_scan_metric.py) instead, same as MaskedFieldMetric - see that
module's docstring for why.
"""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import ClassVar

import pandas as pd
from tqdm import tqdm

from src.data_refinement.card_binder.card_lookup import CardLookup
from src.data_refinement.metrics.version_metadata import (
    MetricVersionMetadata,
    write_dataframe_with_version_metadata,
)
from src.schema.card import GenericCard
from src.schema.game_id import GameId


@dataclass(frozen=True)
class _MaskedFieldRegressionRow:
    """One typed output row: an eligible card, the field masked out of
    it, and its true float value - the same three values scan() writes
    into one parquet row.

    Inputs: none (data holder).
    Output: n/a.
    Side effects: none.
    Exceptions: none.
    """

    nocab_uuid: str
    masked_field: list[str]
    label: float


class MaskedFieldRegressionMetric(ABC):
    """Per-card masked-field regression target: one row per eligible
    card, giving the true float value of a field a Dojo will later
    mask out of that card's raw_content.

    Satisfies the CorpusScanMetric Protocol (corpus_scan_metric.py)
    structurally.
    """

    SOURCE_GAME: ClassVar[GameId]
    MASKED_FIELD: ClassVar[list[str]]
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
                as CardLookup, not the full CardBinder, matching
                MaskedFieldMetric's own constructor rationale.
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
            columns nocab_uuid: str, masked_field: list[str], label:
            float). One row per eligible card. Prints a tqdm progress
            bar to stderr.
        Written with version_metadata.write_dataframe_with_version_metadata()
        so the output carries which self._card_lookup version (see
        CardLookup.version_for()) it was built from, embedded as
        parquet schema metadata - not a sidecar file.

        Exceptions: whatever pyarrow.parquet.write_table raises.

        Example:
            >>> metric = CostRegressionMetric(card_lookup)
            >>> metric.scan()
            PosixPath('data/metrics/dominiontabs/cost_regression.parquet')
        """
        result: list[_MaskedFieldRegressionRow] = []

        # A card with no raw_content at all (the CardBinder "Unknown"
        # sentinel - card_binder.py's ensure_unknown_card()) is
        # excluded here, structurally, rather than leaving every
        # subclass's _is_eligible()/_value_for_card() to each guard
        # against a raw_content lookup that would otherwise KeyError.
        cards = list(self._card_lookup.all_cards(self.SOURCE_GAME))
        for card in tqdm(cards, desc=type(self).__name__, unit="card"):
            if not card.raw_content or not self._is_eligible(card):
                continue
            result.append(self._mask_row(card))

        write_dataframe_with_version_metadata(
            pd.DataFrame([asdict(row) for row in result]),
            self._output_path,
            MetricVersionMetadata(
                game=self.SOURCE_GAME,
                card_binder_version=self._card_lookup.version_for(self.SOURCE_GAME),
            ),
        )
        return self._output_path

    def _mask_row(self, card: GenericCard) -> _MaskedFieldRegressionRow:
        """Build one output row for a single already-eligible card.

        Private helper - single consumer is scan().

        Inputs:
            card: a card _is_eligible() has already accepted.
        Output: a _MaskedFieldRegressionRow - nocab_uuid
            (str(card.nocab_uuid)), masked_field (a copy of
            self.MASKED_FIELD), and label (self._value_for_card(card)).
        Side effects: none.
        Exceptions: whatever _value_for_card() raises.
        """
        return _MaskedFieldRegressionRow(
            nocab_uuid=str(card.nocab_uuid),
            masked_field=list(self.MASKED_FIELD),
            label=self._value_for_card(card),
        )

    def _raw_field_value(self, card: GenericCard) -> str:
        """Walk self.MASKED_FIELD as a path into card.raw_content and
        return the value found there.

        Private helper - shared by every subclass's _value_for_card().
        Deliberately duplicates MaskedFieldMetric._raw_field_value()'s
        identical walk rather than importing it from that module - the
        two classes are siblings, not a hierarchy, and this is the same
        judgment call already made for _MaskedFieldRegressionRow vs.
        masked_field_metric.py's _MaskedFieldRow (see that class's own
        docstring).

        Inputs:
            card: a card to walk raw_content on.
        Output: the raw string value found at self.MASKED_FIELD's path.
        Side effects: none.
        Exceptions: raises KeyError if any path segment is missing
            from card.raw_content.
        """
        node: dict = card.raw_content
        for key in self.MASKED_FIELD[:-1]:
            node = node[key]
        return node[self.MASKED_FIELD[-1]]

    def _is_eligible(self, card: GenericCard) -> bool:
        """Whether card is a valid sample for this metric's masked
        field.

        Default: every card is eligible. Override when a field's raw
        value isn't always parseable as a number (e.g.
        CostRegressionMetric excluding a variable/minimum/uncosted
        cost string) - see this module's docstring for why there's no
        OTHER-bucket fallback for a regression target.

        Inputs:
            card: one card from
                self._card_lookup.all_cards(self.SOURCE_GAME).
        Output: True if card should get an output row.
        Side effects: implementation-defined (expected: none).
        Exceptions: implementation-defined.
        """
        return True

    @abstractmethod
    def _value_for_card(self, card: GenericCard) -> float:
        """The true float value for an already-eligible card.

        Only ever called (via _mask_row()) after _is_eligible(card)
        has returned True. Expected to be implemented via
        self._raw_field_value(card), converted to float as this
        subclass's field requires.

        Inputs:
            card: one eligible card.
        Output: this card's true value for the masked field.
        Side effects: implementation-defined (expected: none).
        Exceptions: implementation-defined.
        """
        raise NotImplementedError
