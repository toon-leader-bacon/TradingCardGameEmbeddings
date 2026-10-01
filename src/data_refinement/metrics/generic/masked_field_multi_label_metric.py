"""Template Method base for multi-label masking metrics: one output row
per (eligible card, masked field), whose label is the SET of
vocabulary members the card has (e.g. an MTG card's colors, a Pokemon's
energy types) rather than exactly one class.

Third sibling of MaskedFieldMetric (masked_field_metric.py) and
MaskedFieldRegressionMetric (masked_field_regression_metric.py): same
scan() sequence and the same _is_eligible()-owns-eligibility rule, but
_labels_for_card() returns a frozenset of LABEL_VALUES members. Kept a
separate class for the reason MaskedFieldRegressionMetric gives: the
row's label dtype differs (list[str] here), and threading it through one
generic base would cost more than the ~20 lines of scan() it saves.

An empty label set is a real label ("none of these", e.g. a colorless
MTG card), not a missing one. A value outside LABEL_VALUES is dropped
rather than bucketed: a multi-label target has no single OTHER slot to
put it in, and the dojo scores each vocabulary member independently.

NOT A Metric[RawRowT] (../metric.py): satisfies CorpusScanMetric
(corpus_scan_metric.py), same as its two siblings.
"""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import ClassVar, Iterable

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
class _MaskedFieldMultiLabelRow:
    """One typed output row: an eligible card, the field masked out of
    it, and its true labels in LABEL_VALUES order.

    Inputs: none (data holder).
    Output: n/a.
    Side effects: none.
    Exceptions: none.
    """

    nocab_uuid: str
    masked_field: list[str]
    label: list[str]


class MaskedFieldMultiLabelMetric(ABC):
    """Per-card masked-field multi-label target: one row per eligible
    card, giving which LABEL_VALUES members the card has for a field a
    Dojo will later mask out of its raw_content.

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
            card_lookup: already-populated registry holding
                self.SOURCE_GAME's cards (a CardBinder satisfies it).
            output_path: overrides DEFAULT_OUTPUT_PATH when given.
        Output: none (constructor).
        Side effects: none - no I/O happens until scan() is called.
        Exceptions: none.
        """
        self._card_lookup = card_lookup
        self._output_path = output_path or self.DEFAULT_OUTPUT_PATH

    def scan(self) -> Path:
        """Walk every self.SOURCE_GAME card, keep the ones _is_eligible()
        accepts, and write one row per eligible card.

        Inputs: none (uses the card_lookup given at construction).
        Output: self._output_path.
        Side effects: creates self._output_path's parent directories if
            missing; writes a parquet file with columns nocab_uuid: str,
            masked_field: list[str], label: list[str] (members of
            LABEL_VALUES, in LABEL_VALUES order; possibly empty), with
            the card binder version embedded as schema metadata. Prints
            a tqdm progress bar to stderr.
        Exceptions: whatever pyarrow.parquet.write_table raises.

        Example:
            >>> ColorsMaskMetric(card_binder).scan()
            PosixPath('data/metrics/scryfall/colors_mask.parquet')
        """
        result: list[_MaskedFieldMultiLabelRow] = []

        # Keep only this subclass's eligible cards; the Unknown sentinel
        # (no raw_content) is never eligible.
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

    def _mask_row(self, card: GenericCard) -> _MaskedFieldMultiLabelRow:
        """Build one output row for an already-eligible card.

        Private helper - single consumer is scan().

        Inputs: card (GenericCard, already eligible).
        Output: a _MaskedFieldMultiLabelRow whose label keeps only
            LABEL_VALUES members of _labels_for_card(card), in
            LABEL_VALUES order.
        Side effects: none.
        Exceptions: whatever _labels_for_card() raises.
        """
        labels = self._labels_for_card(card)
        return _MaskedFieldMultiLabelRow(
            nocab_uuid=str(card.nocab_uuid),
            masked_field=list(self.MASKED_FIELD),
            label=[value for value in self.LABEL_VALUES if value in labels],
        )

    def _raw_field_values(self, card: GenericCard) -> Iterable[str]:
        """Walk self.MASKED_FIELD into card.raw_content and return the
        list found there, or [] if the leaf key is absent (an absent
        list field is the empty set, e.g. a colorless MTG card has no
        "colors" key at all after ingestion drops empty values).

        Private helper - shared by subclasses' _labels_for_card().

        Inputs: card (GenericCard).
        Output: the list at self.MASKED_FIELD's path, or [].
        Side effects: none.
        Exceptions: KeyError if a non-leaf path segment is missing;
            TypeError if the value found is not a list (a str would
            otherwise be read one character at a time).
        """
        node: dict = card.raw_content
        for key in self.MASKED_FIELD[:-1]:
            node = node[key]
        values = node.get(self.MASKED_FIELD[-1], [])
        if not isinstance(values, list):
            raise TypeError(
                f"{self.MASKED_FIELD} of {card.name!r} is not a list: {values!r}"
            )
        return values

    def _is_eligible(self, card: GenericCard) -> bool:
        """Whether card is a valid sample for this metric's field.

        Default: every card is eligible; override where the field does
        not apply to every card.

        Inputs: card (GenericCard).
        Output: True if card should get an output row.
        Side effects: implementation-defined (expected: none).
        Exceptions: implementation-defined.
        """
        return True

    @abstractmethod
    def _labels_for_card(self, card: GenericCard) -> frozenset[str]:
        """The card's true labels, for an already-eligible card.

        Inputs: card (GenericCard, already eligible).
        Output: the LABEL_VALUES members this card has; values outside
            LABEL_VALUES are dropped by _mask_row(); may be empty.
        Side effects: implementation-defined (expected: none).
        Exceptions: implementation-defined.
        """
        raise NotImplementedError
