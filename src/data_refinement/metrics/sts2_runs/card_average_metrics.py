"""Every concrete sts2_runs CardAverageMetric (card_average_metric.py):
sts_gg's nine per-card averages (../sts_gg/card_average_metrics.py) plus
its upgrade rate and act-2 win rate, recomputed from spire_codex +
sts2runs runs.

Each shares its sts_gg counterpart's LABEL_COLUMN by reference, so that
counterpart's dojo wrapper (src/dojos/sts_gg/card_average_dojos.py)
trains on this file unchanged. Unlike sts_gg, these runs include losses,
so the two win rates carry signal here.
"""

from pathlib import Path

from src.data_refinement.metrics.sts2_runs.card_average_metric import (
    CardAverageMetric,
)
from src.data_refinement.metrics.sts2_runs.run_record import (
    CardSlot,
    PlayerRun,
    Sts2Run,
)
from src.data_refinement.metrics.sts_gg import card_average_metrics as sts_gg
from src.data_refinement.metrics.sts_gg.card_upgrade_rate_metric import (
    CardUpgradeRateMetric as StsGgCardUpgradeRateMetric,
)
from src.data_refinement.metrics.sts_gg.card_win_rate_at_act2_metric import (
    CardWinRateAtAct2Metric as StsGgCardWinRateAtAct2Metric,
)

_OUTPUT_DIRECTORY = Path("data/metrics/sts2_runs")


class CardRelicCountMetric(CardAverageMetric):
    """Card -> average relics held at run end."""

    LABEL_COLUMN = sts_gg.CardRelicCountMetric.LABEL_COLUMN
    DEFAULT_OUTPUT_PATH = _OUTPUT_DIRECTORY / "card_relic_count.parquet"

    def _value_for(self, run: Sts2Run, player: PlayerRun, slot: CardSlot) -> float:
        return player.relic_count


class CardTotalDamageTakenMetric(CardAverageMetric):
    """Card -> average damage the player took over the run."""

    LABEL_COLUMN = sts_gg.CardTotalDamageTakenMetric.LABEL_COLUMN
    DEFAULT_OUTPUT_PATH = _OUTPUT_DIRECTORY / "card_total_damage_taken.parquet"

    def _value_for(self, run: Sts2Run, player: PlayerRun, slot: CardSlot) -> float:
        return player.damage_taken


class CardDeckSizeMetric(CardAverageMetric):
    """Card -> average final deck size."""

    LABEL_COLUMN = sts_gg.CardDeckSizeMetric.LABEL_COLUMN
    DEFAULT_OUTPUT_PATH = _OUTPUT_DIRECTORY / "card_deck_size.parquet"

    def _value_for(self, run: Sts2Run, player: PlayerRun, slot: CardSlot) -> float:
        return len(player.deck)


class CardTotalCardsPickedMetric(CardAverageMetric):
    """Card -> average card-reward options the player took."""

    LABEL_COLUMN = sts_gg.CardTotalCardsPickedMetric.LABEL_COLUMN
    DEFAULT_OUTPUT_PATH = _OUTPUT_DIRECTORY / "card_total_cards_picked.parquet"

    def _value_for(self, run: Sts2Run, player: PlayerRun, slot: CardSlot) -> float:
        return player.cards_picked


class CardTotalTurnsMetric(CardAverageMetric):
    """Card -> average combat turns over the run."""

    LABEL_COLUMN = sts_gg.CardTotalTurnsMetric.LABEL_COLUMN
    DEFAULT_OUTPUT_PATH = _OUTPUT_DIRECTORY / "card_total_turns.parquet"

    def _value_for(self, run: Sts2Run, player: PlayerRun, slot: CardSlot) -> float:
        return run.total_turns


class CardElitesKilledMetric(CardAverageMetric):
    """Card -> average elite fights survived."""

    LABEL_COLUMN = sts_gg.CardElitesKilledMetric.LABEL_COLUMN
    DEFAULT_OUTPUT_PATH = _OUTPUT_DIRECTORY / "card_elites_killed.parquet"

    def _value_for(self, run: Sts2Run, player: PlayerRun, slot: CardSlot) -> float:
        return run.elites_killed


class CardFloorsClearedMetric(CardAverageMetric):
    """Card -> average floors cleared."""

    LABEL_COLUMN = sts_gg.CardFloorsClearedMetric.LABEL_COLUMN
    DEFAULT_OUTPUT_PATH = _OUTPUT_DIRECTORY / "card_floors_cleared.parquet"

    def _value_for(self, run: Sts2Run, player: PlayerRun, slot: CardSlot) -> float:
        return run.floors_cleared


class CardTotalCombatsMetric(CardAverageMetric):
    """Card -> average combat rooms entered."""

    LABEL_COLUMN = sts_gg.CardTotalCombatsMetric.LABEL_COLUMN
    DEFAULT_OUTPUT_PATH = _OUTPUT_DIRECTORY / "card_total_combats.parquet"

    def _value_for(self, run: Sts2Run, player: PlayerRun, slot: CardSlot) -> float:
        return run.total_combats


class CardWinRateMetric(CardAverageMetric):
    """Card -> P(win | card in final deck). The naive, survivorship-biased
    rate (see BRAINSTORM.md); CardWinRateAtAct2Metric is the sharper one."""

    LABEL_COLUMN = sts_gg.CardWinRateMetric.LABEL_COLUMN
    DEFAULT_OUTPUT_PATH = _OUTPUT_DIRECTORY / "card_win_rate.parquet"

    def _value_for(self, run: Sts2Run, player: PlayerRun, slot: CardSlot) -> float:
        return run.win


class CardUpgradeRateMetric(CardAverageMetric):
    """Card -> P(copy upgraded by run end | copy in final deck)."""

    LABEL_COLUMN = StsGgCardUpgradeRateMetric.LABEL_COLUMN
    DEFAULT_OUTPUT_PATH = _OUTPUT_DIRECTORY / "card_upgrade_rate.parquet"

    def _value_for(self, run: Sts2Run, player: PlayerRun, slot: CardSlot) -> float:
        return slot.upgraded


class CardWinRateAtAct2Metric(CardAverageMetric):
    """Card -> P(win | copy in the deck as act 2 began), over runs that
    reached act 2.

    "In the deck as act 2 began" is a final-deck copy added before
    Sts2Run.act_2_start_floor (strictly: a copy gained on act 2's first
    floor is not counted, unlike sts_gg's version, which counts it). A
    copy removed before run end is invisible, as in sts_gg.
    """

    LABEL_COLUMN = StsGgCardWinRateAtAct2Metric.LABEL_COLUMN
    DEFAULT_OUTPUT_PATH = _OUTPUT_DIRECTORY / "card_win_rate_at_act2.parquet"

    def _value_for(
        self, run: Sts2Run, player: PlayerRun, slot: CardSlot
    ) -> float | None:
        act_2_start_floor = run.act_2_start_floor
        if act_2_start_floor is None or slot.floor_added >= act_2_start_floor:
            return None
        return run.win
