"""Every concrete sts2_runs DeckLabelMetric (deck_label_metric.py): the
twelve sts_gg deck-label metrics (../sts_gg/deck_label_metrics.py and
../sts_gg/ascension_prediction_metric.py), recomputed from spire_codex +
sts2runs runs.

Each shares its sts_gg counterpart's LABEL_COLUMN and LABEL_TYPE (and
LABEL_VALUES where it has them) by reference, so that counterpart's dojo
wrapper (src/dojos/sts_gg/deck_label_dojos.py) trains on this file
unchanged. Unlike sts_gg, these runs include losses, so WinMetric and
KilledByMetric carry signal here.

Run-level labels (ascension, win, killed_by, turns, combats, elites,
floors) are the same for every player of a co-op run; player-level ones
(character, relics, damage, card picks/skips) are that player's own.
"""

from pathlib import Path

from src.data_refinement.metrics.generic.masked_field_metric import OTHER_LABEL
from src.data_refinement.metrics.sts2_runs.deck_label_metric import DeckLabelMetric
from src.data_refinement.metrics.sts2_runs.run_record import PlayerRun, Sts2Run
from src.data_refinement.metrics.sts_gg import deck_label_metrics as sts_gg
from src.data_refinement.metrics.sts_gg.ascension_prediction_metric import (
    AscensionPredictionMetric as StsGgAscensionPredictionMetric,
)

_OUTPUT_DIRECTORY = Path("data/metrics/sts2_runs")


class AscensionPredictionMetric(DeckLabelMetric):
    """Deck -> ascension level played."""

    LABEL_COLUMN = StsGgAscensionPredictionMetric.LABEL_COLUMN
    LABEL_TYPE = StsGgAscensionPredictionMetric.LABEL_TYPE
    DEFAULT_OUTPUT_PATH = _OUTPUT_DIRECTORY / "ascension_prediction.parquet"

    def _label_for(self, run: Sts2Run, player: PlayerRun) -> int:
        return run.ascension


class CharacterPredictionMetric(DeckLabelMetric):
    """Deck -> the player's character (OTHER_LABEL outside sts_gg's five)."""

    LABEL_COLUMN = sts_gg.CharacterPredictionMetric.LABEL_COLUMN
    LABEL_TYPE = sts_gg.CharacterPredictionMetric.LABEL_TYPE
    LABEL_VALUES = sts_gg.CharacterPredictionMetric.LABEL_VALUES
    DEFAULT_OUTPUT_PATH = _OUTPUT_DIRECTORY / "character_prediction.parquet"

    def _label_for(self, run: Sts2Run, player: PlayerRun) -> str:
        return (
            player.character if player.character in self.LABEL_VALUES else OTHER_LABEL
        )


class WinMetric(DeckLabelMetric):
    """Deck -> won. Final-deck conditioned, so it carries the
    survivorship shortcut BRAINSTORM.md's "Known biases" describes
    (an early loss leaves a near-starter deck)."""

    LABEL_COLUMN = sts_gg.WinMetric.LABEL_COLUMN
    LABEL_TYPE = sts_gg.WinMetric.LABEL_TYPE
    DEFAULT_OUTPUT_PATH = _OUTPUT_DIRECTORY / "win.parquet"

    def _label_for(self, run: Sts2Run, player: PlayerRun) -> bool:
        return run.win


class KilledByMetric(DeckLabelMetric):
    """Deck -> the encounter that ended a lost run (sts_gg's closed
    vocabulary, OTHER_LABEL outside it). Won runs write no row."""

    LABEL_COLUMN = sts_gg.KilledByMetric.LABEL_COLUMN
    LABEL_TYPE = sts_gg.KilledByMetric.LABEL_TYPE
    LABEL_VALUES = sts_gg.KilledByMetric.LABEL_VALUES
    DEFAULT_OUTPUT_PATH = _OUTPUT_DIRECTORY / "killed_by.parquet"

    def _label_for(self, run: Sts2Run, player: PlayerRun) -> str | None:
        if run.killed_by is None:
            return None
        return run.killed_by if run.killed_by in self.LABEL_VALUES else OTHER_LABEL


class RelicCountMetric(DeckLabelMetric):
    """Deck -> relics held at run end."""

    LABEL_COLUMN = sts_gg.RelicCountMetric.LABEL_COLUMN
    LABEL_TYPE = sts_gg.RelicCountMetric.LABEL_TYPE
    DEFAULT_OUTPUT_PATH = _OUTPUT_DIRECTORY / "relic_count.parquet"

    def _label_for(self, run: Sts2Run, player: PlayerRun) -> int:
        return player.relic_count


class TotalDamageTakenMetric(DeckLabelMetric):
    """Deck -> damage the player took over the run."""

    LABEL_COLUMN = sts_gg.TotalDamageTakenMetric.LABEL_COLUMN
    LABEL_TYPE = sts_gg.TotalDamageTakenMetric.LABEL_TYPE
    DEFAULT_OUTPUT_PATH = _OUTPUT_DIRECTORY / "total_damage_taken.parquet"

    def _label_for(self, run: Sts2Run, player: PlayerRun) -> int:
        return player.damage_taken


class TotalCardsPickedMetric(DeckLabelMetric):
    """Deck -> card-reward options the player took."""

    LABEL_COLUMN = sts_gg.TotalCardsPickedMetric.LABEL_COLUMN
    LABEL_TYPE = sts_gg.TotalCardsPickedMetric.LABEL_TYPE
    DEFAULT_OUTPUT_PATH = _OUTPUT_DIRECTORY / "total_cards_picked.parquet"

    def _label_for(self, run: Sts2Run, player: PlayerRun) -> int:
        return player.cards_picked


class TotalCardsSkippedMetric(DeckLabelMetric):
    """Deck -> card-reward options the player passed on."""

    LABEL_COLUMN = sts_gg.TotalCardsSkippedMetric.LABEL_COLUMN
    LABEL_TYPE = sts_gg.TotalCardsSkippedMetric.LABEL_TYPE
    DEFAULT_OUTPUT_PATH = _OUTPUT_DIRECTORY / "total_cards_skipped.parquet"

    def _label_for(self, run: Sts2Run, player: PlayerRun) -> int:
        return player.cards_skipped


class TotalTurnsMetric(DeckLabelMetric):
    """Deck -> combat turns over the run."""

    LABEL_COLUMN = sts_gg.TotalTurnsMetric.LABEL_COLUMN
    LABEL_TYPE = sts_gg.TotalTurnsMetric.LABEL_TYPE
    DEFAULT_OUTPUT_PATH = _OUTPUT_DIRECTORY / "total_turns.parquet"

    def _label_for(self, run: Sts2Run, player: PlayerRun) -> int:
        return run.total_turns


class ElitesKilledMetric(DeckLabelMetric):
    """Deck -> elite fights survived."""

    LABEL_COLUMN = sts_gg.ElitesKilledMetric.LABEL_COLUMN
    LABEL_TYPE = sts_gg.ElitesKilledMetric.LABEL_TYPE
    DEFAULT_OUTPUT_PATH = _OUTPUT_DIRECTORY / "elites_killed.parquet"

    def _label_for(self, run: Sts2Run, player: PlayerRun) -> int:
        return run.elites_killed


class FloorsClearedMetric(DeckLabelMetric):
    """Deck -> floors cleared (Sts2Run.floors_cleared)."""

    LABEL_COLUMN = sts_gg.FloorsClearedMetric.LABEL_COLUMN
    LABEL_TYPE = sts_gg.FloorsClearedMetric.LABEL_TYPE
    DEFAULT_OUTPUT_PATH = _OUTPUT_DIRECTORY / "floors_cleared.parquet"

    def _label_for(self, run: Sts2Run, player: PlayerRun) -> int:
        return run.floors_cleared


class TotalCombatsMetric(DeckLabelMetric):
    """Deck -> combat rooms entered."""

    LABEL_COLUMN = sts_gg.TotalCombatsMetric.LABEL_COLUMN
    LABEL_TYPE = sts_gg.TotalCombatsMetric.LABEL_TYPE
    DEFAULT_OUTPUT_PATH = _OUTPUT_DIRECTORY / "total_combats.parquet"

    def _label_for(self, run: Sts2Run, player: PlayerRun) -> int:
        return run.total_combats
