"""Streaming metric: a guide's deck -> its community votes
(BRAINSTORM.md multi-card #11, "Guide Vote Prediction from Deck").

LABEL: signed_log_votes = sign(v) * ln(1 + |v|). Votes are net
(up minus down) and heavy-tailed: over the 60,197 guides they run from
-120 to 1,594 (median 1, 23k zeros, 5,268 negative, p99 38). log(1 + v)
is undefined for v <= -1, so the signed log keeps the sign and
compresses both tails (1,594 -> 7.37, -120 -> -4.80). The regression
cell z-scores it from there.

BIASES: votes measure how the guide (its write-up, its author's reach,
its age: older guides had longer to collect votes) was received, not
only how strong the deck is; read the label as "deck reception".

Rows point into the published Gwent box (deck_uuid, see
published_guide_decks.py), so the output is flagged requires_deck_box
and a dojo checks that box's binder version. Nothing is written to any
deck box.
"""

import math
from pathlib import Path
from typing import ClassVar

import pyarrow as pa

from src.data_refinement.card_binder.card_lookup import CardLookup
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.metrics.parquet_builder import ParquetBuilder
from src.data_refinement.metrics.play_gwent.published_guide_decks import (
    PublishedGuideDecks,
)
from src.data_refinement.metrics.version_metadata import (
    MetricVersionMetadata,
    schema_with_version_metadata,
)
from src.schema.game_id import GameId


def signed_log_votes(votes: int) -> float:
    """sign(votes) * ln(1 + |votes|).

    Inputs: votes (int, may be negative). Output: float.
    Side effects: none. Exceptions: none.

    Example:
        >>> round(signed_log_votes(-120), 2)
        -4.8
    """
    return math.copysign(math.log1p(abs(votes)), votes)


class GuideVotesMetric:
    """Guide deck -> (guide_id, deck_uuid, signed_log_votes), one row per
    guide whose deck is in the published box. Satisfies Metric[dict]
    (../metric.py) structurally."""

    LABEL_COLUMN: ClassVar[str] = "signed_log_votes"
    DEFAULT_OUTPUT_PATH: ClassVar[Path] = Path(
        "data/metrics/play_gwent/guide_votes.parquet"
    )

    def __init__(
        self,
        card_lookup: CardLookup,
        deck_box: DeckBox,
        output_path: Path | None = None,
    ) -> None:
        """
        Inputs:
            card_lookup: the Gwent binder; only its version and the
                leader lookup are read.
            deck_box: the published Gwent deck box; only read.
            output_path: overrides DEFAULT_OUTPUT_PATH.
        Output: none (constructor).
        Side effects: creates output_path's parent directory; opens
            output_path for writing (truncating it) until finalize().
        Exceptions: whatever ParquetBuilder raises opening output_path.
        """
        self._guide_decks = PublishedGuideDecks(deck_box, card_lookup)
        self._output_path = output_path or self.DEFAULT_OUTPUT_PATH
        schema = schema_with_version_metadata(
            pa.schema(
                [
                    ("guide_id", pa.int64()),
                    ("deck_uuid", pa.string()),
                    (self.LABEL_COLUMN, pa.float64()),
                ]
            ),
            MetricVersionMetadata(
                game=GameId.GWENT,
                card_binder_version=card_lookup.version_for(GameId.GWENT),
                requires_deck_box=True,
            ),
        )
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = ParquetBuilder(self._output_path, schema)

    def accumulate(self, row: dict) -> None:
        """Buffer row's guide as one output row, if its deck is published.

        Inputs: row (one parsed guides.jsonl object).
        Output: none.
        Side effects: reads the box; buffers one row in the writer.
        Exceptions: KeyError if row has no "id".

        Example:
            >>> metric.accumulate(json.loads(line))
        """
        guide_deck = self._guide_decks.guide_deck_for_row(row)
        if guide_deck is None:
            return
        self._writer.write_row(
            {
                "guide_id": guide_deck.guide_id,
                "deck_uuid": str(guide_deck.deck_uuid),
                self.LABEL_COLUMN: signed_log_votes(guide_deck.votes),
            }
        )

    def finalize(self) -> Path:
        """Flush and close the writer. Idempotent.

        Inputs: none. Output: the output path.
        Side effects: writes the remaining rows and the parquet footer.
        Exceptions: whatever ParquetBuilder.close() raises.

        Example:
            >>> metric.finalize()
            PosixPath('data/metrics/play_gwent/guide_votes.parquet')
        """
        self._writer.close()
        return self._output_path
