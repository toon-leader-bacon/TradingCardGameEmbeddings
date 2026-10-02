"""Single-card inclusion-rate metrics over playgwent.com guide decks
(BRAINSTORM.md single-card #1 and #3), the Gwent twins of
../fabtcg_decklists/card_inclusion_metrics.py.

  - CardInclusionRateMetric: P(card in deck | card is legal for the
    deck's faction), pooled over every faction.
  - FactionConditionedInclusionMetric: P(card in deck | faction), one
    rate per GWENT_FACTIONS faction, with a 0 deck count where the card
    is illegal for that faction (a masked position, not a 0 rate).

Both are accumulators over raw guide rows (scanner.py): each guide's
deck is read from the published box (PublishedGuideDecks, read-only)
and counted in a FactionInclusionTally; finalize() writes one row per
card. Inclusion is only counted against legal factions, so a rate never
exceeds 1. Card universe: cards held by at least one guide deck,
leaders excluded (they define the faction; LeaderMaskedFromDeckMetric
covers them). Known bias: guides span 2019-2026 and every patch, so an
old card's rate includes years a newer card did not exist for.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

import pyarrow as pa

from src.data_refinement.card_binder.card_lookup import CardLookup
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.metrics.parquet_builder import ParquetBuilder
from src.data_refinement.metrics.play_gwent.faction_inclusion_tally import (
    FactionInclusion,
    FactionInclusionTally,
)
from src.data_refinement.metrics.play_gwent.published_guide_decks import (
    GWENT_FACTIONS,
    PublishedGuideDecks,
)
from src.data_refinement.metrics.version_metadata import (
    MetricVersionMetadata,
    schema_with_version_metadata,
)
from src.schema.game_id import GameId

_OUTPUT_DIRECTORY = Path("data/metrics/play_gwent")

# A pooled rate over fewer legal decks than this is too noisy to train on
MIN_LEGAL_DECKS = 20


class _FactionInclusionMetric(ABC):
    """Shared accumulate/finalize for the two metrics below (Template
    Method): a subclass names its label fields and builds one card's
    row. Satisfies Metric[dict] (../metric.py) structurally."""

    DEFAULT_OUTPUT_PATH: ClassVar[Path]

    def __init__(
        self,
        card_lookup: CardLookup,
        deck_box: DeckBox,
        output_path: Path | None = None,
    ) -> None:
        """
        Inputs:
            card_lookup: the Gwent binder; never written to.
            deck_box: the published Gwent deck box; only read.
            output_path: overrides DEFAULT_OUTPUT_PATH.
        Output: none (constructor).
        Side effects: none (the output file is only opened by
            finalize()).
        Exceptions: none.
        """
        self._card_lookup = card_lookup
        self._guide_decks = PublishedGuideDecks(deck_box, card_lookup)
        self._tally = FactionInclusionTally()
        self._output_path = output_path or self.DEFAULT_OUTPUT_PATH

    def accumulate(self, row: dict) -> None:
        """Count row's published guide deck, if it has one.

        Inputs: row (one parsed guides.jsonl object).
        Output: none.
        Side effects: reads the box; updates the tally.
        Exceptions: KeyError if row has no "id".

        Example:
            >>> metric.accumulate(json.loads(line))
        """
        guide_deck = self._guide_decks.guide_deck_for_row(row)
        if guide_deck is not None:
            self._tally.add(guide_deck)

    def finalize(self) -> Path:
        """Write one row per card in the tally's universe that the
        subclass keeps.

        Inputs: none. Output: the output path.
        Side effects: creates the output directory; writes the parquet
            (truncating it), stamped with the Gwent binder version.
        Exceptions: whatever ParquetBuilder raises.

        Example:
            >>> metric.finalize()
            PosixPath('data/metrics/play_gwent/card_inclusion_rate.parquet')
        """
        schema = schema_with_version_metadata(
            pa.schema([("nocab_uuid", pa.string()), *self._label_fields()]),
            MetricVersionMetadata(
                game=GameId.GWENT,
                card_binder_version=self._card_lookup.version_for(GameId.GWENT),
            ),
        )
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = ParquetBuilder(self._output_path, schema)
        for card in self._tally.included_cards(self._card_lookup):
            label_values = self._label_values(self._tally.legal_inclusions(card))
            if label_values is not None:
                writer.write_row({"nocab_uuid": str(card.nocab_uuid), **label_values})
        writer.close()
        return self._output_path

    @abstractmethod
    def _label_fields(self) -> list[tuple[str, pa.DataType]]:
        """The output columns after nocab_uuid.

        Inputs: none. Output: [(column name, pyarrow type)].
        Side effects: none. Exceptions: none.
        """
        raise NotImplementedError

    @abstractmethod
    def _label_values(
        self, inclusions: list[FactionInclusion]
    ) -> dict[str, Any] | None:
        """One card's label columns, or None to write no row.

        Inputs: inclusions (the card's counts per legal faction).
        Output: {column: value} for _label_fields(), or None.
        Side effects: none. Exceptions: none.
        """
        raise NotImplementedError


class CardInclusionRateMetric(_FactionInclusionMetric):
    """Card -> P(card in deck | card legal for the deck's faction), over
    every guide deck. Cards legal for fewer than MIN_LEGAL_DECKS decks
    get no row."""

    LABEL_COLUMN: ClassVar[str] = "inclusion_rate"
    DEFAULT_OUTPUT_PATH: ClassVar[Path] = (
        _OUTPUT_DIRECTORY / "card_inclusion_rate.parquet"
    )

    def _label_fields(self) -> list[tuple[str, pa.DataType]]:
        return [
            (self.LABEL_COLUMN, pa.float64()),
            ("legal_deck_count", pa.int64()),
            ("included_deck_count", pa.int64()),
        ]

    def _label_values(
        self, inclusions: list[FactionInclusion]
    ) -> dict[str, Any] | None:
        legal = sum(inclusion.legal_deck_count for inclusion in inclusions)
        if legal < MIN_LEGAL_DECKS:
            return None
        included = sum(inclusion.included_deck_count for inclusion in inclusions)
        return {
            self.LABEL_COLUMN: included / legal,
            "legal_deck_count": legal,
            "included_deck_count": included,
        }


class FactionConditionedInclusionMetric(_FactionInclusionMetric):
    """Card -> [P(card in deck | faction f) for f in GWENT_FACTIONS].

    Two parallel list columns in GWENT_FACTIONS order:
    inclusion_rate_by_faction (None where the count is 0) and
    legal_deck_count_by_faction (the faction's deck count where the card
    is legal for it, else 0). A neutral card fills all six positions, a
    faction card one (two for a dual-faction card). The dojo masks
    positions under its own minimum count.
    """

    LABEL_VALUES: ClassVar[tuple[str, ...]] = GWENT_FACTIONS
    DEFAULT_OUTPUT_PATH: ClassVar[Path] = (
        _OUTPUT_DIRECTORY / "faction_conditioned_inclusion.parquet"
    )

    def _label_fields(self) -> list[tuple[str, pa.DataType]]:
        return [
            ("inclusion_rate_by_faction", pa.list_(pa.float64())),
            ("legal_deck_count_by_faction", pa.list_(pa.int64())),
        ]

    def _label_values(
        self, inclusions: list[FactionInclusion]
    ) -> dict[str, Any] | None:
        by_faction = {inclusion.faction: inclusion for inclusion in inclusions}
        counts = [
            by_faction[name].legal_deck_count if name in by_faction else 0
            for name in GWENT_FACTIONS
        ]
        if not any(counts):
            return None
        rates = [
            by_faction[name].included_deck_count / count if count else None
            for name, count in zip(GWENT_FACTIONS, counts)
        ]
        return {
            "inclusion_rate_by_faction": rates,
            "legal_deck_count_by_faction": counts,
        }
