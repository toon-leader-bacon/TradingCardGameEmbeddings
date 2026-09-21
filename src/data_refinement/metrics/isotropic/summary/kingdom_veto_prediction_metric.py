"""Streaming metric: BRAINSTORM.md multi-card #3 - given the full
undealt candidate kingdom (board.supply union vetoed), predict which
card(s) got vetoed.

CANDIDATE POOL AS A DECKBOX GROUP: the pre-veto candidate pool isn't a
deck (it's never bought, played, or scored) - stored via deck_box per
the metric_writing skill's "this deck box... may write ANY multi-card
collection... hands, draft packs, partial decks etc" allowance, using
GenericDeck purely as a generic "named card group" container. Content-
addressed the same way a real deck is (hash_utils.deck_uuid_from_cards)
so two games with the same candidate pool collapse together.

VARIABLE-SET LABEL: LABEL_VALUES is not fixed here (unlike e.g.
../../generic/deck_card_mask_metric.py consumers) - the vetoed card(s)
range over dominiontabs' whole card pool, so the label column is a
list of card uuids, not a single fixed-vocabulary string. A game with
an empty vetoed list is skipped entirely (a real "no label" outcome,
not an all-zero row) - see accumulate()'s docstring.
"""

import logging
from pathlib import Path
from typing import ClassVar
from uuid import UUID

import pyarrow as pa
import pyarrow.parquet as pq

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.metrics.hash_utils import deck_uuid_from_cards
from src.data_refinement.metrics.isotropic.summary.row_utils import (
    card_uuid_for_name,
    is_natural_kingdom,
    kingdom_card_names,
)
from src.schema.card import GenericDeck
from src.schema.game_id import GameId

_logger = logging.getLogger(__name__)


class KingdomVetoPredictionMetric:
    """(candidate_pool_uuid, vetoed_card_uuids), one row per game with
    at least one real veto, written as soon as accumulate() sees it.

    Satisfies the Metric[dict] Protocol (../../metric.py) structurally.
    """

    DEFAULT_OUTPUT_PATH: ClassVar[Path] = Path(
        "data/metrics/isotropic/kingdom_veto_prediction.parquet"
    )

    def __init__(
        self,
        card_binder: CardBinder,
        deck_box: DeckBox,
        output_path: Path | None = None,
    ) -> None:
        """
        Inputs:
            card_binder: registry to resolve isotropic card names
                against - must already have dominiontabs' cards
                ingested (this class never writes to it).
            deck_box: the metrics-private DeckBox every candidate pool
                this metric sees is written into - see class docstring.
            output_path: overrides DEFAULT_OUTPUT_PATH when given.
        Output: none (constructor).
        Side effects: creates output_path's parent directories if
            missing; opens output_path for writing (truncating any
            existing file) via a pyarrow.parquet.ParquetWriter held
            open for the lifetime of this instance - callers MUST call
            finalize() when done, or the file is left incomplete.
        Exceptions: whatever pyarrow.parquet.ParquetWriter raises on
            failure to open output_path for writing.
        """
        self._card_binder = card_binder
        self._deck_box = deck_box
        self._output_path = output_path or self.DEFAULT_OUTPUT_PATH
        self._output_schema = pa.schema(
            [
                ("candidate_pool_uuid", pa.string()),
                ("vetoed_card_uuids", pa.list_(pa.string())),
            ]
        )
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = pq.ParquetWriter(self._output_path, self._output_schema)
        self._closed = False

    def accumulate(self, row: dict) -> None:
        """Write one (candidate_pool_uuid, vetoed_card_uuids) row for
        one natural-kingdom game with at least one real veto.

        Inputs:
            row: one parsed Flavor A summary row, carrying "board"
                (with "supply") and, optionally, "vetoed".
        Output: none.
        Side effects: writes exactly one row to the open ParquetWriter
            when row has a non-empty "vetoed" list AND is a natural
            kingdom (row_utils.is_natural_kingdom()) - writes nothing
            for a row with no vetoes (a real "nothing to predict"
            outcome, not an error) or a generator-constrained kingdom
            (BRAINSTORM.md's known-biases note: a curated kingdom's
            veto behavior isn't a fair sample). Writes the candidate
            pool (board.supply union vetoed) into self._deck_box via
            create_if_absent(), but only when a row also produces an
            output row - an un-vetoed pool is never a useful training
            input for this metric, so it's not worth minting. Emits one
            logging.error() per vetoed/candidate-pool card name that
            fails to resolve.
        Exceptions: none expected beyond whatever row_utils' own
            functions raise for a malformed row.

        Example:
            >>> metric = KingdomVetoPredictionMetric(card_binder, deck_box)
            >>> metric.accumulate(row)
            >>> metric.finalize()
        """
        if not is_natural_kingdom(row):
            return

        vetoed_names = row.get("vetoed", [])
        if not vetoed_names:
            return

        vetoed_uuids: list[UUID] = []
        for name in vetoed_names:
            card_uuid = card_uuid_for_name(self._card_binder, name)
            if card_uuid is None:
                _logger.error(
                    "KingdomVetoPredictionMetric: unresolved vetoed card "
                    "name %r - excluding it from this row's label",
                    name,
                )
                continue
            vetoed_uuids.append(card_uuid)
        if not vetoed_uuids:
            # Every vetoed name failed to resolve - no real label left
            # to write, so the candidate pool isn't worth minting
            # either (see class docstring's "only when a row also
            # produces an output row" note).
            return

        kingdom_names = kingdom_card_names(row)
        pool_deck = self._candidate_pool_deck(kingdom_names, vetoed_names)
        self._deck_box.create_if_absent(pool_deck)

        output_row = pa.Table.from_pydict(
            {
                "candidate_pool_uuid": [str(pool_deck.nocab_uuid)],
                "vetoed_card_uuids": [[str(uuid) for uuid in vetoed_uuids]],
            },
            schema=self._output_schema,
        )
        self._writer.write_table(output_row)

    def finalize(self) -> Path:
        """Close the underlying ParquetWriter.

        Does NOT save self._deck_box - that's the calling driver's own
        responsibility, since the box is shared across metrics.

        Inputs: none.
        Output: self._output_path.
        Side effects: closes the ParquetWriter opened in __init__, if
            not already closed.
        Exceptions: whatever ParquetWriter.close() raises.

        Example:
            >>> metric.finalize()
            PosixPath('data/metrics/isotropic/kingdom_veto_prediction.parquet')
        """
        if not self._closed:
            self._writer.close()
            self._closed = True
        return self._output_path

    def _candidate_pool_deck(
        self, kingdom_names: list[str], vetoed_names: list[str]
    ) -> GenericDeck:
        """Build the GenericDeck for one game's pre-veto candidate pool.

        Private helper - single consumer is accumulate().

        Inputs:
            kingdom_names: this game's dealt board.supply.
            vetoed_names: this game's vetoed card names.
        Output: a GenericDeck over the resolved union of both name
            lists (duplicates collapsed - this is a set of distinct
            candidate cards, not a multiset with meaningful copy
            counts, unlike a real deck), source_game=GameId.DOMINION.
            provenance is left as its default (None) - this is a
            private, metrics-only DeckBox entry, not a canonical
            ingested deck, mirroring ../../sts_gg/deck_label_metric.py's
            own GenericDeck construction (which likewise omits
            provenance entirely).
        Side effects: none (the caller writes it into deck_box).
        Exceptions: none.
        """
        pool_names = sorted(set(kingdom_names) | set(vetoed_names))
        card_nocab_uuids: list[UUID] = []
        for name in pool_names:
            card_uuid = card_uuid_for_name(self._card_binder, name)
            if card_uuid is None:
                _logger.error(
                    "KingdomVetoPredictionMetric: unresolved candidate-pool "
                    "card name %r - excluding it from this pool",
                    name,
                )
                continue
            card_nocab_uuids.append(card_uuid)
        deck_uuid = deck_uuid_from_cards(card_nocab_uuids)
        return GenericDeck(
            nocab_uuid=deck_uuid,
            source_game=GameId.DOMINION,
            name="isotropic pre-veto candidate pool",
            card_nocab_uuids=card_nocab_uuids,
        )
