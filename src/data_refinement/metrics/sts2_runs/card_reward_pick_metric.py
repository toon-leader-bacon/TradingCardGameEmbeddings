"""CardRewardPickMetric - BRAINSTORM.md #15 (Draft Pick Prediction): given
the deck so far and the cards a reward offered, which one was taken, or
none.

A streaming Metric[Sts2Run] (../metric.py). One output row per card-reward
choice a scored player faced (CardRewardChoice, run_record.py): about 12
rows per run, so the scan writes on the order of 10^7 rows. The writer
streams row groups (ParquetBuilder) and nothing is held back, so memory
stays flat; a dojo reads the file in chunks.

Output columns:
    run_id, deck_uuid: the run and the player's published deck box deck;
        run_id is the dojo's split group (a run yields many rows, and
        splitting by row would put one run in both TRAIN and TEST).
    floor: the reward's floor.
    deck_uuids: the deck on arrival, one card uuid per copy. An unordered
        multiset: its order carries no meaning.
    offered_uuids: the cards offered, in the order shown.
    picked_uuid: the offered card taken; NULL means the player skipped.
Unlike requires_deck_box metrics, the deck is stored inline (a partial deck
is not a published-box deck), so only the CardBinder version is stamped.

ROW POLICY: a card with no spire_codex alias cannot be embedded. In the
deck it is dropped (it is context). In the offer, or as the pick, it would
corrupt the position label, so that whole choice writes no row.
"""

from pathlib import Path

import pyarrow as pa

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.metrics.parquet_builder import ParquetBuilder
from src.data_refinement.metrics.sts2_runs.run_record import (
    CardRewardChoice,
    PlayerRun,
    Sts2Run,
)
from src.data_refinement.metrics.version_metadata import (
    MetricVersionMetadata,
    schema_with_version_metadata,
)
from src.schema.game_id import GameId

_UUID_LIST = pa.list_(pa.string())
# Rows buffered per row group: ~28 uuid strings each, so ~40 MB at most.
_BATCH_SIZE = 10_000


class CardRewardPickMetric:
    """(deck so far, cards offered) -> the card taken, or none.

    Satisfies Metric[Sts2Run] (../metric.py) structurally.
    """

    DEFAULT_OUTPUT_PATH = Path("data/metrics/sts2_runs/card_reward_pick.parquet")

    def __init__(
        self, card_binder: CardBinder, output_path: Path | None = None
    ) -> None:
        """
        Inputs:
            card_binder: the Slay the Spire 2 binder; only its version is
                read (stamped into the output).
            output_path: overrides DEFAULT_OUTPUT_PATH.
        Output: none (constructor).
        Side effects: creates output_path's parent directory; opens
            output_path for writing (truncating it) until finalize().
        Exceptions: whatever ParquetBuilder raises opening output_path.
        """
        self._output_path = output_path or self.DEFAULT_OUTPUT_PATH
        schema = schema_with_version_metadata(
            self._output_schema(),
            MetricVersionMetadata(
                game=GameId.SLAY_THE_SPIRE_2,
                card_binder_version=card_binder.version_for(GameId.SLAY_THE_SPIRE_2),
            ),
        )
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = ParquetBuilder(self._output_path, schema, _BATCH_SIZE)

    def accumulate(self, run: Sts2Run) -> None:
        """Write one row per usable card-reward choice of run's players.

        Inputs: run (Sts2Run), already filtered by the scanner
            (scanner.py's scored_run()).
        Output: none.
        Side effects: buffers rows in the open ParquetBuilder.
        Exceptions: whatever the writer raises.

        Example:
            >>> metric = CardRewardPickMetric(binder, scratch_path)
            >>> metric.accumulate(run)
            >>> metric.finalize()
        """
        # One row per usable choice of each player
        for player in run.players:
            for choice in player.card_rewards:
                row = self._row_or_none(run, player, choice)
                if row is None:
                    continue
                self._writer.write_row(row)

    def finalize(self) -> Path:
        """Flush and close the writer. Idempotent.

        Inputs: none. Output: the output path.
        Side effects: writes the remaining rows and the parquet footer.
        Exceptions: whatever ParquetBuilder.close() raises.

        Example:
            >>> metric.finalize()
            PosixPath('data/metrics/sts2_runs/card_reward_pick.parquet')
        """
        self._writer.close()
        return self._output_path

    @staticmethod
    def _row_or_none(
        run: Sts2Run, player: PlayerRun, choice: CardRewardChoice
    ) -> dict[str, object] | None:
        """The output row for one choice, or None if the ROW POLICY
        drops it (an offered or picked card has no alias).

        Inputs: run, player (the choice's owner), choice.
        Output: dict keyed by every output column, or None.
        Side effects: none. Exceptions: none.
        """
        if any(card_uuid is None for card_uuid in choice.offered):
            return None
        picked = (
            None if choice.picked_index is None else choice.offered[choice.picked_index]
        )
        return {
            "run_id": run.run_id,
            "deck_uuid": str(player.deck_uuid),
            "floor": choice.floor,
            "deck_uuids": [
                str(uuid) for uuid in choice.deck_before if uuid is not None
            ],
            "offered_uuids": [str(uuid) for uuid in choice.offered if uuid is not None],
            "picked_uuid": None if picked is None else str(picked),
        }

    @staticmethod
    def _output_schema() -> pa.Schema:
        """The output schema, without version metadata.

        Inputs: none. Output: pa.Schema. Side effects: none.
        Exceptions: none.
        """
        return pa.schema(
            [
                ("run_id", pa.string()),
                ("deck_uuid", pa.string()),
                ("floor", pa.int32()),
                ("deck_uuids", _UUID_LIST),
                ("offered_uuids", _UUID_LIST),
                ("picked_uuid", pa.string()),
            ]
        )
