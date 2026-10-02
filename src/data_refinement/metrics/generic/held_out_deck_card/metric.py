"""HeldOutDeckCardMetric: hold one distinct card out of a published
deck and ask which of K + 1 same-game candidates it was.

A ClassVar-configured base satisfying CorpusScanMetric (scan() -> Path).
Driven from a published DeckBox (data/final/decks/<game>.db), not from
raw rows: unlike DeckCardMaskMetric, whose target is a role ("the
leader") the box no longer knows, a held-out card needs only the
deck's multiset. A subclass fixes ClassVars only: SOURCE_GAME,
DEFAULT_OUTPUT_PATH and SAMPLING.

Output columns: deck_uuid (str, the published box's deck uuid),
target_card_uuid (str), candidate_uuids (list<str>, target + decoys,
shuffled). Version metadata: the lookup's binder version,
requires_deck_box=True. The box is only read.
"""

from abc import ABC
from pathlib import Path
from typing import ClassVar

import pyarrow as pa

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.card_binder.card_lookup import CardLookup
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.metrics.generic.held_out_deck_card.candidate_sampler import (
    CandidateSampler,
    HeldOutCardRow,
)
from src.data_refinement.metrics.generic.held_out_deck_card.deck_sample import (
    DeckSample,
)
from src.data_refinement.metrics.generic.held_out_deck_card.sampling import (
    HeldOutCardSampling,
)
from src.data_refinement.metrics.parquet_builder import ParquetBuilder
from src.data_refinement.metrics.version_metadata import (
    MetricVersionMetadata,
    schema_with_version_metadata,
)
from src.schema.game_id import GameId


class HeldOutDeckCardMetric(ABC):
    """One published deck box -> (deck, held-out card, candidates) rows.

    Subclasses set SOURCE_GAME, DEFAULT_OUTPUT_PATH and SAMPLING.
    """

    SOURCE_GAME: ClassVar[GameId]
    DEFAULT_OUTPUT_PATH: ClassVar[Path]
    SAMPLING: ClassVar[HeldOutCardSampling]

    def __init__(
        self,
        card_lookup: CardLookup,
        deck_box: DeckBox,
        output_path: Path | None = None,
        sampling: HeldOutCardSampling | None = None,
    ) -> None:
        """
        Inputs:
            card_lookup: SOURCE_GAME's cards; only its version is read.
            deck_box: the published box for SOURCE_GAME; only read.
            output_path: overrides DEFAULT_OUTPUT_PATH.
            sampling: overrides SAMPLING (e.g. a small max_decks for a
                subset run).
        Output: none (constructor).
        Side effects: none (nothing is opened until scan()).
        Exceptions: none.
        """
        self._card_lookup = card_lookup
        self._deck_box = deck_box
        self._output_path = output_path or self.DEFAULT_OUTPUT_PATH
        self._sampling = sampling or self.SAMPLING

    def scan(self) -> Path:
        """Sample the box, draw targets and decoys, write the parquet.

        Inputs: none.
        Output: the path written.
        Side effects: reads deck_box; creates the output's parent
            directory; overwrites the output file.
        Exceptions: ValueError if deck_box's stamped binder version for
            SOURCE_GAME is missing or differs from card_lookup's (the
            rows would reference a different card set), or if the
            sample holds no decks.

        Example:
            >>> GwentHeldOutCardMetric(binder, box).scan()
            PosixPath('data/metrics/final_decks/held_out_card_gwent.parquet')
        """
        # Validate inputs: the box must match the binder we stamp
        card_binder_version = self._card_lookup.version_for(self.SOURCE_GAME)
        self._require_current_deck_box(card_binder_version)

        # Sample decks once, then draw every deck's rows
        sample = DeckSample.from_deck_box(
            self._deck_box,
            self.SOURCE_GAME,
            self._sampling.max_decks,
            self._sampling.seed,
            CardBinder.unknown_card_uuid(self.SOURCE_GAME),
        )
        sampler = CandidateSampler(sample, self._sampling)

        # Stream rows to parquet
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = ParquetBuilder(self._output_path, self._schema(card_binder_version))
        try:
            for deck_index in range(sample.deck_count):
                for row in sampler.rows_for(deck_index):
                    writer.write_row(_output_row(row))
        finally:
            writer.close()
        return self._output_path

    def _require_current_deck_box(self, card_binder_version: str) -> None:
        """Raise unless deck_box was stamped with card_binder_version
        for SOURCE_GAME.

        Inputs: card_binder_version (the lookup's).
        Output: none. Side effects: none.
        Exceptions: ValueError on a missing or different stamp.
        """
        stamped = self._deck_box.card_binder_version_for(self.SOURCE_GAME)
        if stamped != card_binder_version:
            raise ValueError(
                f"{type(self).__name__}: the deck box's {self.SOURCE_GAME.value} decks "
                f"were minted from CardBinder version {stamped!r}, but the card "
                f"lookup is at {card_binder_version!r}; re-ingest the deck box first"
            )

    def _schema(self, card_binder_version: str) -> pa.Schema:
        """The output schema with version metadata (requires_deck_box).

        Inputs: card_binder_version. Output: pa.Schema.
        Side effects: none. Exceptions: none.
        """
        return schema_with_version_metadata(
            pa.schema(
                [
                    ("deck_uuid", pa.string()),
                    ("target_card_uuid", pa.string()),
                    ("candidate_uuids", pa.list_(pa.string())),
                ]
            ),
            MetricVersionMetadata(
                game=self.SOURCE_GAME,
                card_binder_version=card_binder_version,
                requires_deck_box=True,
            ),
        )


def _output_row(row: HeldOutCardRow) -> dict[str, object]:
    """One HeldOutCardRow as a parquet row of strings.

    Inputs: row. Output: dict with deck_uuid, target_card_uuid,
        candidate_uuids. Side effects: none. Exceptions: none.
    """
    return {
        "deck_uuid": str(row.deck_uuid),
        "target_card_uuid": str(row.target_card_uuid),
        "candidate_uuids": [str(card) for card in row.candidate_uuids],
    }
