"""Accumulation metric: the per-card character distribution - for each
card, P(character == c | card in final deck) for every character c
that card was ever seen paired with.

NOT A CardAverageMetric: unlike card_average_metrics.py's nine
siblings, a card's output here isn't one scalar but a whole
distribution (one row per (card, character) pair actually observed),
so this doesn't fit CardAverageMetric's Template Method (a single
running sum/count per card) - reusing that base over a genuinely
different output shape would be forcing an abstraction over only
superficial similarity. The per-copy iteration and card-id resolution
rule are still duplicated from it directly, per this container's
established convention of duplicating that small rule per file rather
than centralizing it (see card_upgrade_rate_metric.py's module
docstring).

deck_label_metrics.py's CharacterPredictionMetric is the deck-input
mirror of this exact prediction task (predict character from a whole
deck, one row per run); this is the same task's single-card, per-card-
frequency-table version instead.
"""

import logging
from pathlib import Path
from typing import ClassVar
from uuid import UUID

import pandas as pd

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox
from src.schema.data_source import DataSource
from src.schema.game_id import GameId

_logger = logging.getLogger(__name__)

_STS_GG_CARD_ID_PREFIX = "CARD."


class CardCharacterPredictionMetric:
    """Per-card character frequency table: P(character | card in final
    deck), one row per (card, character) combination actually seen.

    Satisfies the Metric[dict] Protocol (../metric.py) structurally.
    accumulate() only tallies in-memory per-(card, character) counts;
    finalize() does the real per-card division and write.
    """

    DEFAULT_OUTPUT_PATH: ClassVar[Path] = Path(
        "data/metrics/sts_gg/card_character_prediction.parquet"
    )

    def __init__(
        self,
        card_binder: CardBinder,
        deck_box: DeckBox | None = None,
        output_path: Path | None = None,
    ) -> None:
        """
        Inputs:
            card_binder: registry to resolve sts_gg's "CARD.<id>"
                references against - must already have spire_codex's
                cards ingested (this class never writes to it).
            deck_box: accepted only for constructor-shape consistency
                with this container's deck-input metrics - see
                ../README.md. This metric's output is per-card, not
                per-deck, so it never reads from or writes into
                deck_box.
            output_path: overrides DEFAULT_OUTPUT_PATH when given.
        Output: none (constructor).
        Side effects: none - no I/O happens until finalize() is
            called.
        Exceptions: none.
        """
        self._card_binder = card_binder
        self._deck_box = deck_box
        self._output_path = output_path or self.DEFAULT_OUTPUT_PATH
        self._total_count: dict[UUID, int] = {}
        self._character_count: dict[tuple[UUID, str], int] = {}

    def accumulate(self, row: dict) -> None:
        """Tally every card copy in one run's final deck toward this
        metric's running per-(card, character) counts.

        Inputs:
            row: one parsed sts_gg run, carrying at least "character"
                (str, e.g. "CHARACTER.SILENT") and "deck" (list of
                {"id": str, ...} card entries - the run's final deck).
        Output: none.
        Side effects: updates self._total_count/_character_count in
            place, once per physical copy in row["deck"] - see
            card_upgrade_rate_metric.py's module docstring's COPY
            COUNTING section for the same rule applied here. Emits one
            logging.error() per deck entry that fails to resolve.
        Exceptions: raises if row is missing "character" or "deck".

        Example:
            >>> metric = CardCharacterPredictionMetric(card_binder)
            >>> metric.accumulate(json.loads(next(runs_file)))
            >>> metric.finalize()
        """
        character = row["character"]

        for card_entry in row["deck"]:
            card_uuid = self._card_uuid(card_entry["id"])
            if card_uuid is None:
                _logger.error(
                    "CardCharacterPredictionMetric: unresolved card id %r - "
                    "excluding it from this run's tally",
                    card_entry["id"],
                )
                continue

            self._total_count[card_uuid] = self._total_count.get(card_uuid, 0) + 1
            key = (card_uuid, character)
            self._character_count[key] = self._character_count.get(key, 0) + 1

    def finalize(self) -> Path:
        """Compute every seen (card, character) pair's probability and
        write one row per pair to self._output_path.

        Inputs: none (uses accumulated state).
        Output: self._output_path.
        Side effects: creates self._output_path's parent directories
            if missing; writes self._output_path (a parquet file with
            columns nocab_uuid: str, character: str, probability:
            float, sample_count: int - sample_count is the CARD's
            total count, shared across every character row for that
            card, so every row for one card sums its probability
            column to 1.0).
        Exceptions: whatever pandas.DataFrame.to_parquet raises.

        Example:
            >>> metric.finalize()
            PosixPath('data/metrics/sts_gg/card_character_prediction.parquet')
        """
        result: list[dict] = [
            self._character_row(card_uuid, character)
            for (card_uuid, character) in self._character_count
        ]

        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(result).to_parquet(self._output_path, index=False)
        return self._output_path

    def _character_row(self, card_uuid: UUID, character: str) -> dict:
        """Build one output row for a single already-tallied
        (card, character) pair.

        Private helper - single consumer is finalize().

        Inputs:
            card_uuid: a key already present in self._total_count.
            character: a character already tallied for card_uuid.
        Output: a dict with keys "nocab_uuid" (str), "character" (str),
            "probability" (float, this pair's count divided by
            card_uuid's total count), and "sample_count" (int,
            card_uuid's total count).
        Side effects: none.
        Exceptions: none.
        """
        total = self._total_count[card_uuid]
        count = self._character_count[(card_uuid, character)]
        return {
            "nocab_uuid": str(card_uuid),
            "character": character,
            "probability": count / total,
            "sample_count": total,
        }

    def _card_uuid(self, raw_card_id: str) -> UUID | None:
        """Look up the nocab_uuid for one sts_gg deck entry's card id.

        Private helper - single consumer is accumulate(). Same
        "CARD." prefix-strip + alias lookup rule as every other metric
        in this container.

        Inputs:
            raw_card_id: one deck entry's "id" field, e.g.
                "CARD.GRAND_FINALE".
        Output: the matching nocab_uuid, or None if unresolved.
        Side effects: none.
        Exceptions: none.
        """
        stripped_id = raw_card_id.removeprefix(_STS_GG_CARD_ID_PREFIX)
        card = self._card_binder.get_by_alias(
            GameId.SLAY_THE_SPIRE_2, DataSource.SPIRE_CODEX, stripped_id
        )
        return card.nocab_uuid if card is not None else None
