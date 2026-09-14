"""Concrete PackCardTallyMetric (pack_card_tally_metric.py) subclasses:
three of round 1's four single-card metrics from
src/data_refinement/metrics/seventeenlands/draft_data/BRAINSTORM.md's
"Human Review Short List" - see plans/draft_data_metrics.md's
"Component overview" #4 for the key-shape rationale behind each.

PickNumberDecayCurveMetric (this list's fourth single-card metric) also
subclasses PackCardTallyMetric, but lives in its own file
(pick_number_decay_curve_metric.py) since its finalize() is overridden
entirely - see that module's docstring.
"""

from pathlib import Path
from typing import ClassVar
from uuid import UUID

from src.data_refinement.metrics.seventeenlands.draft_data.pack_card_tally_metric import (
    PackCardTallyMetric,
)

_DEFAULT_OUTPUT_DIR = Path("data/metrics/seventeenlands/draft_data")


class CardTakeRateMetric(PackCardTallyMetric):
    """P(picked | in pack, pick_number, pack_number) - BRAINSTORM.md's
    single-card metric #1 (Card Take Rate)."""

    KEY_COLUMNS: ClassVar[tuple[str, ...]] = ("pack_number", "pick_number")
    DEFAULT_OUTPUT_PATH = _DEFAULT_OUTPUT_DIR / "card_take_rate.parquet"

    def _tally_key(self, row: dict, card_uuid: UUID) -> tuple:
        """See PackCardTallyMetric._tally_key(). Key = (card_uuid,
        row["pack_number"], row["pick_number"])."""
        return (card_uuid, row["pack_number"], row["pick_number"])


class FirstPickRateMetric(PackCardTallyMetric):
    """P(pick == card | pack_number == 0, pick_number == 0, card in
    pack) - BRAINSTORM.md's single-card metric #4 (First-Pick Rate /
    P1P1 Take Rate).

    KEY_COLUMNS is empty - _is_eligible_row()'s restriction to pack 0 /
    pick 0 already does all of this metric's conditioning, so its key
    is the card alone.
    """

    KEY_COLUMNS: ClassVar[tuple[str, ...]] = ()
    DEFAULT_OUTPUT_PATH = _DEFAULT_OUTPUT_DIR / "first_pick_rate.parquet"

    def _is_eligible_row(self, row: dict) -> bool:
        """See PackCardTallyMetric._is_eligible_row(). True only when
        row["pack_number"] == 0 and row["pick_number"] == 0."""
        return row["pack_number"] == 0 and row["pick_number"] == 0

    def _tally_key(self, row: dict, card_uuid: UUID) -> tuple:
        """See PackCardTallyMetric._tally_key(). Key = (card_uuid,) -
        no extra dimensions, see class docstring."""
        return (card_uuid,)


class RankStratifiedTakeRateMetric(PackCardTallyMetric):
    """Card Take Rate, additionally stratified by rank bucket -
    BRAINSTORM.md's single-card metric #6 (Rank-Stratified Take Rate).

    KEY_COLUMNS below keeps pack_number/pick_number alongside rank, per
    plans/draft_data_metrics.md's "Open questions" #1 - a
    (card, rank)-only key (dropping pack_number/pick_number) is the
    documented alternative if this proves too sparse in practice.
    """

    KEY_COLUMNS: ClassVar[tuple[str, ...]] = ("pack_number", "pick_number", "rank")
    DEFAULT_OUTPUT_PATH = _DEFAULT_OUTPUT_DIR / "rank_stratified_take_rate.parquet"

    def _tally_key(self, row: dict, card_uuid: UUID) -> tuple:
        """See PackCardTallyMetric._tally_key(). Key = (card_uuid,
        row["pack_number"], row["pick_number"], row["rank"])."""
        return (card_uuid, row["pack_number"], row["pick_number"], row["rank"])
