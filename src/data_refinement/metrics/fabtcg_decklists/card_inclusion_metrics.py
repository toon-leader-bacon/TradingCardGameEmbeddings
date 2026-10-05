"""Single-card inclusion-rate metrics over published FaB decklists
(BRAINSTORM.md single-card #1 and #2).

  - CardInclusionRateMetric: P(card in deck | card is legal for the
    deck's hero), pooled over every hero.
  - HeroConditionedInclusionMetric: P(card in deck | hero), one rate per
    HERO_NAMES hero, with a 0 deck count where the card is illegal for
    that hero (a masked position, not a 0 rate).

Both are accumulators over raw decklist rows {"slug": str} (scanner.py):
each row's deck is read from the published box (PublishedDecklists,
read-only) and counted in a HeroInclusionTally; finalize() writes one
row per card. Legality is hero_legality.is_legal_for_hero. A card's
inclusion is only counted against heroes it is legal for, so a rate
never exceeds 1 (the 0.19% of slots that break the rule are dropped).

Card universe: cards held by at least one deck (hero_inclusion_tally.py
says why). Known bias: the deck box resolves a decklist name printed in
several pitches to one printing, so a rate is really per card name.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, NamedTuple

import pyarrow as pa

from src.data_refinement.card_binder.card_lookup import CardLookup
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.metrics.fabtcg_decklists.hero_inclusion_tally import (
    HeroInclusion,
    HeroInclusionTally,
)
from src.data_refinement.metrics.fabtcg_decklists.hero_labels import HERO_NAMES
from src.data_refinement.metrics.fabtcg_decklists.published_decklists import (
    PublishedDecklists,
)
from src.data_refinement.metrics.parquet_builder import ParquetBuilder
from src.data_refinement.metrics.version_metadata import (
    MetricVersionMetadata,
    schema_with_version_metadata,
)
from src.schema.game_id import GameId

_OUTPUT_DIRECTORY = Path("data/metrics/fabtcg_decklists")

# A pooled rate over fewer legal decks than this is too noisy to train on
MIN_LEGAL_DECKS = 20


class _CardInclusionRateRow(NamedTuple):
    """CardInclusionRateMetric._label_values()'s own shape - one field
    per column its _label_fields() lists, in the same order. Field
    names match those column names exactly, so as_dict() needs no
    renaming."""

    inclusion_rate: float
    legal_deck_count: int
    included_deck_count: int

    def as_dict(self) -> dict[str, Any]:
        """{column: value} for ParquetBuilder.write_row() - the one
        place this typed row is ever turned back into a dict.
        Inputs: none. Output: dict[str, Any]. Side effects: none.
        Exceptions: none."""
        return self._asdict()


class _HeroConditionedInclusionRow(NamedTuple):
    """HeroConditionedInclusionMetric._label_values()'s own shape - one
    field per column its _label_fields() lists, in the same order.
    Field names match those column names exactly, so as_dict() needs no
    renaming."""

    inclusion_rate_by_hero: list[float | None]
    legal_deck_count_by_hero: list[int]

    def as_dict(self) -> dict[str, Any]:
        """{column: value} for ParquetBuilder.write_row() - the one
        place this typed row is ever turned back into a dict.
        Inputs: none. Output: dict[str, Any]. Side effects: none.
        Exceptions: none."""
        return self._asdict()


_LabelValues = _CardInclusionRateRow | _HeroConditionedInclusionRow


class _HeroInclusionMetric(ABC):
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
            card_lookup: the FaB binder; never written to.
            deck_box: the published FaB deck box; only read.
            output_path: overrides DEFAULT_OUTPUT_PATH.
        Output: none (constructor).
        Side effects: reads every FaB deck in deck_box once. The output
            file is only opened by finalize().
        Exceptions: none.
        """
        self._card_lookup = card_lookup
        self._decklists = PublishedDecklists(deck_box, card_lookup)
        self._tally = HeroInclusionTally()
        self._output_path = output_path or self.DEFAULT_OUTPUT_PATH

    def accumulate(self, row: dict) -> None:
        """Count row's published decklist, if it has one.

        Inputs: row ({"slug": str}, one raw decklist file).
        Output: none.
        Side effects: updates the tally.
        Exceptions: KeyError if row has no "slug".

        Example:
            >>> metric.accumulate({"slug": "gabe-sher-lexi-deck-calling-antwerp"})
        """
        decklist = self._decklists.decklist_for_slug(row["slug"])
        if decklist is not None:
            self._tally.add(decklist)

    def finalize(self) -> Path:
        """Write one row per card in the tally's universe that the
        subclass keeps.

        Inputs: none. Output: the output path.
        Side effects: creates the output directory; writes the parquet
            (truncating it), stamped with the FaB binder version.
        Exceptions: whatever ParquetBuilder raises.

        Example:
            >>> metric.finalize()
            PosixPath('data/metrics/fabtcg_decklists/card_inclusion_rate.parquet')
        """
        schema = schema_with_version_metadata(
            pa.schema([("nocab_uuid", pa.string()), *self._label_fields()]),
            MetricVersionMetadata(
                game=GameId.FLESH_AND_BLOOD,
                card_binder_version=self._card_lookup.version_for(
                    GameId.FLESH_AND_BLOOD
                ),
            ),
        )
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = ParquetBuilder(self._output_path, schema)
        for card in self._tally.included_cards(self._card_lookup):
            label_values = self._label_values(self._tally.legal_inclusions(card))
            if label_values is not None:
                writer.write_row(
                    {"nocab_uuid": str(card.nocab_uuid), **label_values.as_dict()}
                )
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
    def _label_values(self, inclusions: list[HeroInclusion]) -> _LabelValues | None:
        """One card's label columns, or None to write no row.

        Inputs: inclusions (the card's counts per legal hero).
        Output: the typed row matching _label_fields(), or None.
        Side effects: none. Exceptions: none.
        """
        raise NotImplementedError


class CardInclusionRateMetric(_HeroInclusionMetric):
    """Card -> P(card in deck | card legal for the deck's hero), over
    every single-hero published decklist. Cards legal for fewer than
    MIN_LEGAL_DECKS decks get no row."""

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
        self, inclusions: list[HeroInclusion]
    ) -> _CardInclusionRateRow | None:
        legal = sum(inclusion.legal_deck_count for inclusion in inclusions)
        if legal < MIN_LEGAL_DECKS:
            return None
        included = sum(inclusion.included_deck_count for inclusion in inclusions)
        return _CardInclusionRateRow(
            inclusion_rate=included / legal,
            legal_deck_count=legal,
            included_deck_count=included,
        )


class HeroConditionedInclusionMetric(_HeroInclusionMetric):
    """Card -> [P(card in deck | hero h) for h in HERO_NAMES].

    Two parallel list columns in HERO_NAMES order: inclusion_rate_by_hero
    (None where the count is 0) and legal_deck_count_by_hero (the hero's
    deck count where the card is legal for it, else 0). A card legal for
    no HERO_NAMES hero gets no row. The dojo masks positions under its
    own minimum count.
    """

    LABEL_VALUES: ClassVar[tuple[str, ...]] = HERO_NAMES
    DEFAULT_OUTPUT_PATH: ClassVar[Path] = (
        _OUTPUT_DIRECTORY / "hero_conditioned_inclusion.parquet"
    )

    def _label_fields(self) -> list[tuple[str, pa.DataType]]:
        return [
            ("inclusion_rate_by_hero", pa.list_(pa.float64())),
            ("legal_deck_count_by_hero", pa.list_(pa.int64())),
        ]

    def _label_values(
        self, inclusions: list[HeroInclusion]
    ) -> _HeroConditionedInclusionRow | None:
        by_hero = {inclusion.hero_name: inclusion for inclusion in inclusions}
        counts = [
            by_hero[name].legal_deck_count if name in by_hero else 0
            for name in HERO_NAMES
        ]
        if not any(counts):
            return None
        rates = [
            by_hero[name].included_deck_count / count if count else None
            for name, count in zip(HERO_NAMES, counts)
        ]
        return _HeroConditionedInclusionRow(
            inclusion_rate_by_hero=rates, legal_deck_count_by_hero=counts
        )
