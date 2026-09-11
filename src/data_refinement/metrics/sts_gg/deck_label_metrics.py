"""Every concrete DeckLabelMetric (deck_label_metric.py) this container
has today - eleven metrics from BRAINSTORM.md's multi-card section (#7,
#9) and its stats-block-derived siblings, all sharing DeckLabelMetric's
Template Method: only LABEL_COLUMN/LABEL_TYPE/DEFAULT_OUTPUT_PATH and
_label_for_run() differ per class; deck resolution, hashing, DeckBox
writes, and output-row writes are all inherited unchanged.

FIELD SOURCE: every label below is a top-level or "stats"-nested field
already present on a run's raw row (see BRAINSTORM.md's "What the raw
data actually looks like") - none of these condition on a floor/act
boundary the way card_win_rate_at_act2_metric.py does, since every
label here is a whole-run scalar, not a value tied to a specific point
in the run.

WARNING: KilledBy seems to be only null for the current data source.
So it's not a good metric to use.
NULLABLE LABEL: KilledByMetric's label is None on a win (killed_by is
only ever set on a loss) - pyarrow's string type is nullable by
default, so this is written as a real null, not a sentinel string.
"""

from pathlib import Path
from typing import ClassVar

import pyarrow as pa

from src.data_refinement.metrics.masked_field_metric import OTHER_LABEL
from src.data_refinement.metrics.sts_gg.deck_label_metric import DeckLabelMetric


class RelicCountMetric(DeckLabelMetric):
    """Deck -> relicCount (BRAINSTORM.md multi-card #9).

    BRAINSTORM.md frames this against a deck snapshot at a fixed floor
    (to pair relicCount from that same floor, not the run's final
    total) - this class uses the final deck/final relicCount instead,
    the same simplification WinMetric/FloorsClearedMetric make below,
    not the sharper floor-conditioned version.
    """

    LABEL_COLUMN = "relic_count"
    LABEL_TYPE = pa.int64()
    DEFAULT_OUTPUT_PATH = Path("data/metrics/sts_gg/relic_count.parquet")

    def _label_for_run(self, row: dict) -> int:
        return row["relicCount"]


class CharacterPredictionMetric(DeckLabelMetric):
    """Deck -> character (BRAINSTORM.md multi-card #7).

    Label is the raw "CHARACTER.<id>" string as sts_gg writes it (e.g.
    "CHARACTER.SILENT") - unlike a card id, this is never looked up
    against a CardBinder, so there's no reason to strip its prefix.

    LABEL_VALUES, confirmed against data/raw/sts_gg/runs.jsonl (full
    925-row scan, 2026-09-10): exactly five distinct "character" values
    appear - DEFECT, IRONCLAD, NECROBINDER, REGENT, SILENT - no others.
    OTHER_LABEL is included anyway (masked_field_metric.OTHER_LABEL,
    shared sentinel - see gwent_one's masked-field metrics for the same
    convention) as a safety net for a playable character added after
    this list was written, not because one was observed; see
    _label_for_run()'s fallback.
    """

    LABEL_COLUMN = "character"
    LABEL_TYPE = pa.string()
    DEFAULT_OUTPUT_PATH = Path("data/metrics/sts_gg/character_prediction.parquet")
    LABEL_VALUES: ClassVar[tuple[str, ...]] = (
        "CHARACTER.DEFECT",
        "CHARACTER.IRONCLAD",
        "CHARACTER.NECROBINDER",
        "CHARACTER.REGENT",
        "CHARACTER.SILENT",
        OTHER_LABEL,
    )

    def _label_for_run(self, row: dict) -> str:
        character = row["character"]
        if character not in self.LABEL_VALUES:
            return OTHER_LABEL
        return character


class TotalDamageTakenMetric(DeckLabelMetric):
    """Deck -> stats.totalDamageTaken."""

    LABEL_COLUMN = "total_damage_taken"
    LABEL_TYPE = pa.int64()
    DEFAULT_OUTPUT_PATH = Path("data/metrics/sts_gg/total_damage_taken.parquet")

    def _label_for_run(self, row: dict) -> int:
        return row["stats"]["totalDamageTaken"]


class TotalCardsPickedMetric(DeckLabelMetric):
    """Deck -> stats.totalCardsPicked."""

    LABEL_COLUMN = "total_cards_picked"
    LABEL_TYPE = pa.int64()
    DEFAULT_OUTPUT_PATH = Path("data/metrics/sts_gg/total_cards_picked.parquet")

    def _label_for_run(self, row: dict) -> int:
        return row["stats"]["totalCardsPicked"]


class TotalCardsSkippedMetric(DeckLabelMetric):
    """Deck -> stats.totalCardsSkipped."""

    LABEL_COLUMN = "total_cards_skipped"
    LABEL_TYPE = pa.int64()
    DEFAULT_OUTPUT_PATH = Path("data/metrics/sts_gg/total_cards_skipped.parquet")

    def _label_for_run(self, row: dict) -> int:
        return row["stats"]["totalCardsSkipped"]


class TotalTurnsMetric(DeckLabelMetric):
    """Deck -> stats.totalTurns."""

    LABEL_COLUMN = "total_turns"
    LABEL_TYPE = pa.int64()
    DEFAULT_OUTPUT_PATH = Path("data/metrics/sts_gg/total_turns.parquet")

    def _label_for_run(self, row: dict) -> int:
        return row["stats"]["totalTurns"]


class ElitesKilledMetric(DeckLabelMetric):
    """Deck -> stats.elitesKilled."""

    LABEL_COLUMN = "elites_killed"
    LABEL_TYPE = pa.int64()
    DEFAULT_OUTPUT_PATH = Path("data/metrics/sts_gg/elites_killed.parquet")

    def _label_for_run(self, row: dict) -> int:
        return row["stats"]["elitesKilled"]


class FloorsClearedMetric(DeckLabelMetric):
    """Deck -> stats.floorsCleared (BRAINSTORM.md multi-card #8's raw
    field, deck-input framed here as the final deck rather than the
    starting deck)."""

    LABEL_COLUMN = "floors_cleared"
    LABEL_TYPE = pa.int64()
    DEFAULT_OUTPUT_PATH = Path("data/metrics/sts_gg/floors_cleared.parquet")

    def _label_for_run(self, row: dict) -> int:
        return row["stats"]["floorsCleared"]


class TotalCombatsMetric(DeckLabelMetric):
    """Deck -> stats.totalCombats."""

    LABEL_COLUMN = "total_combats"
    LABEL_TYPE = pa.int64()
    DEFAULT_OUTPUT_PATH = Path("data/metrics/sts_gg/total_combats.parquet")

    def _label_for_run(self, row: dict) -> int:
        return row["stats"]["totalCombats"]


class KilledByMetric(DeckLabelMetric):
    """
    WARNING: KilledBy seems to be only null for the current data
    source. So it's not a good metric to use.

    Deck -> killedBy (BRAINSTORM.md multi-group #1's per-run
    analogue - null on a win, an encounter/event id string on a loss).
    """

    LABEL_COLUMN = "killed_by"
    LABEL_TYPE = pa.string()
    DEFAULT_OUTPUT_PATH = Path("data/metrics/sts_gg/killed_by.parquet")

    def _label_for_run(self, row: dict) -> str | None:
        return row["killedBy"]


class WinMetric(DeckLabelMetric):
    """Deck -> win - the naive, unconditioned baseline BRAINSTORM.md's
    "Known biases" section already flags as contaminated by
    survivorship (an early-abandoned run's final deck looks close to
    its starting deck) - kept anyway as the simplest possible
    deck-outcome baseline; card_win_rate_at_act2_metric.py's act-2
    conditioning is this container's sharper alternative, not a
    replacement for this one.
    """

    LABEL_COLUMN = "win"
    LABEL_TYPE = pa.bool_()
    DEFAULT_OUTPUT_PATH = Path("data/metrics/sts_gg/win.parquet")

    def _label_for_run(self, row: dict) -> bool:
        return row["win"]
