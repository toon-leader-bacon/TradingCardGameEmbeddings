"""Every concrete CardAverageMetric (card_average_metric.py) this
container has today - the per-card mirror of deck_label_metrics.py's
per-deck metrics: same nine run-level scalars, now averaged per card
across every run that card appeared in, rather than referenced once
per run via a deck_uuid.

CardWinRateMetric is this container's naive, per-card,
unconditioned win rate - P(win | card in final deck) - distinct from
card_win_rate_at_act2_metric.py's CardWinRateAtAct2Metric, which
conditions on the card being present as of act 2's start specifically
to sidestep the exact survivorship contamination this naive version
carries (see BRAINSTORM.md's "Known biases" section, and
deck_label_metrics.py's WinMetric docstring for the deck-level
analogue of this same caveat).
"""

from pathlib import Path

from src.data_refinement.metrics.sts_gg.card_average_metric import CardAverageMetric


class CardRelicCountMetric(CardAverageMetric):
    """Card -> average relicCount, across every run it appeared in."""

    LABEL_COLUMN = "average_relic_count"
    DEFAULT_OUTPUT_PATH = Path("data/metrics/sts_gg/card_relic_count.parquet")

    def _value_for_run(self, row: dict) -> int:
        return row["relicCount"]


class CardTotalDamageTakenMetric(CardAverageMetric):
    """Card -> average stats.totalDamageTaken."""

    LABEL_COLUMN = "average_total_damage_taken"
    DEFAULT_OUTPUT_PATH = Path("data/metrics/sts_gg/card_total_damage_taken.parquet")

    def _value_for_run(self, row: dict) -> int:
        return row["stats"]["totalDamageTaken"]


class CardDeckSizeMetric(CardAverageMetric):
    """Card -> average deckSize (the run's final deck size, cards
    removed during the run included in the count via deckSize's own
    definition, not deck_label_metrics.py's card-list length)."""

    LABEL_COLUMN = "average_deck_size"
    DEFAULT_OUTPUT_PATH = Path("data/metrics/sts_gg/card_deck_size.parquet")

    def _value_for_run(self, row: dict) -> int:
        return row["deckSize"]


class CardTotalCardsPickedMetric(CardAverageMetric):
    """Card -> average stats.totalCardsPicked (unlike deckSize, never
    reduced by mid-run removals - see module docstring's sibling
    CardDeckSizeMetric)."""

    LABEL_COLUMN = "average_total_cards_picked"
    DEFAULT_OUTPUT_PATH = Path("data/metrics/sts_gg/card_total_cards_picked.parquet")

    def _value_for_run(self, row: dict) -> int:
        return row["stats"]["totalCardsPicked"]


class CardTotalTurnsMetric(CardAverageMetric):
    """Card -> average stats.totalTurns."""

    LABEL_COLUMN = "average_total_turns"
    DEFAULT_OUTPUT_PATH = Path("data/metrics/sts_gg/card_total_turns.parquet")

    def _value_for_run(self, row: dict) -> int:
        return row["stats"]["totalTurns"]


class CardElitesKilledMetric(CardAverageMetric):
    """Card -> average stats.elitesKilled."""

    LABEL_COLUMN = "average_elites_killed"
    DEFAULT_OUTPUT_PATH = Path("data/metrics/sts_gg/card_elites_killed.parquet")

    def _value_for_run(self, row: dict) -> int:
        return row["stats"]["elitesKilled"]


class CardFloorsClearedMetric(CardAverageMetric):
    """Card -> average stats.floorsCleared."""

    LABEL_COLUMN = "average_floors_cleared"
    DEFAULT_OUTPUT_PATH = Path("data/metrics/sts_gg/card_floors_cleared.parquet")

    def _value_for_run(self, row: dict) -> int:
        return row["stats"]["floorsCleared"]


class CardTotalCombatsMetric(CardAverageMetric):
    """Card -> average stats.totalCombats."""

    LABEL_COLUMN = "average_total_combats"
    DEFAULT_OUTPUT_PATH = Path("data/metrics/sts_gg/card_total_combats.parquet")

    def _value_for_run(self, row: dict) -> int:
        return row["stats"]["totalCombats"]


class CardWinRateMetric(CardAverageMetric):
    """Card -> P(win | card in final deck) - see module docstring."""

    LABEL_COLUMN = "win_rate"
    DEFAULT_OUTPUT_PATH = Path("data/metrics/sts_gg/card_win_rate.parquet")

    def _value_for_run(self, row: dict) -> bool:
        return row["win"]
