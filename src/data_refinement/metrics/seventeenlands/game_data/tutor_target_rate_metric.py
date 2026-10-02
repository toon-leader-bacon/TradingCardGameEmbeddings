"""TutorTargetRateMetric - BRAINSTORM.md's single-card metric "Tutor
Target Rate": P(card in tutored_<name> | card in deck_<name>) - among
games where a card was in the deck, how often did a tutor effect
actually fetch it that game.

A vectorized Metric[GameDataChunk] (see game_data/README.md).
Per deck column it counts (games in deck, games in deck and
tutored) in a CardColumnTallies; "tutored" is the deck column's card
present under any tutored_<name> column, via ZoneCounts.present_for().

Structurally close to draft_data's take-rate shape (a ratio of two
per-card tallies, draft_data/pack_card_tally_metric.py's
PackCardTallyMetric) but NOT a subclass of it: that class is specific
to draft_data's pack/pick concept, and reaching into draft_data/ for a
base would be the cross-cousin-directory import PRINCIPLES.md flags.

tutored_<name> is rare (~8.5% of games have any hit at all per
BRAINSTORM.md) - sample_count is written alongside the rate so a
consumer can filter out noisy near-zero estimates from cards seen in
few games, per BRAINSTORM.md's own flag for this metric.
"""

from pathlib import Path
from typing import ClassVar
from uuid import UUID

import numpy as np
import pandas as pd

from src.data_refinement.metrics.seventeenlands.game_data.card_column_tallies import (
    CardColumnTallies,
)
from src.data_refinement.metrics.seventeenlands.game_data.game_data_chunk import (
    GameDataChunk,
    GameZone,
)
from src.data_refinement.metrics.version_metadata import (
    MetricVersionMetadata,
    write_dataframe_with_version_metadata,
)

_DEFAULT_OUTPUT_PATH = Path(
    "data/metrics/seventeenlands/game_data/tutor_target_rate.parquet"
)

# Per deck column: games with the card in the deck, and of those, games
# it was also tutored.
_IN_DECK, _TUTORED = range(2)
_TALLY_COUNT = 2


class TutorTargetRateMetric:
    """Card -> P(tutored | in deck).

    Satisfies the Metric[GameDataChunk] Protocol (../../metric.py)
    structurally.
    """

    DEFAULT_OUTPUT_PATH: ClassVar[Path] = _DEFAULT_OUTPUT_PATH

    def __init__(
        self,
        version_metadata: MetricVersionMetadata,
        output_path: Path | None = None,
    ) -> None:
        """Start a metric with no tallies.

        Inputs:
            version_metadata: the CardBinder version this run reads,
                stamped onto the output.
            output_path: overrides DEFAULT_OUTPUT_PATH when given.
        Output: none (constructor).
        Side effects: none (no I/O until finalize()).
        Exceptions: none.
        """
        self._version_metadata = version_metadata
        self._output_path = output_path or self.DEFAULT_OUTPUT_PATH
        self._tallies = CardColumnTallies(type(self).__name__, _TALLY_COUNT, np.int64)

    def accumulate(self, chunk: GameDataChunk) -> None:
        """Count, per deck column, the games its card was in the deck
        and the games it was also tutored.

        Inputs: chunk.
        Output: none.
        Side effects: adds this chunk to the per-column tallies.
        Exceptions: ValueError if chunk's deck columns differ from the
            first chunk's (chunks from two CSVs).

        Example:
            >>> metric = TutorTargetRateMetric(version_metadata)
            >>> metric.accumulate(chunk)
            >>> metric.finalize()
        """
        deck = chunk.zones[GameZone.DECK]
        in_deck = deck.present()

        # Was each deck column's card tutored, under any tutored column?
        tutored = chunk.zones[GameZone.TUTORED].present_for(deck.card_uuids)

        increments = np.stack(
            [
                in_deck.sum(axis=0, dtype=np.int64),
                (in_deck & tutored).sum(axis=0, dtype=np.int64),
            ]
        )
        self._tallies.add(deck.card_uuids, increments)

    def finalize(self) -> Path:
        """Compute every seen card's tutor target rate and write one
        row per card to self._output_path.

        Inputs: none (uses accumulated state).
        Output: self._output_path.
        Side effects: creates self._output_path's parent directories if
            missing; writes self._output_path (a parquet file with
            columns nocab_uuid: str, tutor_target_rate: float,
            sample_count: int - one row per card seen in deck_<name> at
            least once). Built from row dicts, as the row
            implementation did.
        Exceptions: whatever pyarrow.parquet.write_table raises.

        Example:
            >>> metric.finalize()
            PosixPath('data/metrics/seventeenlands/game_data/tutor_target_rate.parquet')
        """
        result: list[dict] = []

        # One output row per card in at least one deck
        for card_uuid, tallies in self._tallies.per_card().items():
            if tallies[_IN_DECK] == 0:
                continue
            result.append(
                _tutor_rate_row(
                    card_uuid, int(tallies[_IN_DECK]), int(tallies[_TUTORED])
                )
            )

        write_dataframe_with_version_metadata(
            pd.DataFrame(result), self._output_path, self._version_metadata
        )
        return self._output_path


def _tutor_rate_row(card_uuid: UUID, times_in_deck: int, times_tutored: int) -> dict:
    """One output row: nocab_uuid, tutor_target_rate, sample_count.

    Inputs: card_uuid, times_in_deck (>= 1), times_tutored.
    Output: dict keyed by the output columns.
    Side effects: none. Exceptions: none.
    """
    return {
        "nocab_uuid": str(card_uuid),
        "tutor_target_rate": times_tutored / times_in_deck,
        "sample_count": times_in_deck,
    }
