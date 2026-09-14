"""Template Method base for accumulation metrics that tally
(times_in_pack, times_picked) per (card_uuid, *stratifying key), across
every pack_card_<name> column present (> 0) on a row - see
plans/draft_data_metrics.md's "Component overview" #4.

Three of round 1's four single-card metrics
(pack_card_tally_metrics.py's CardTakeRateMetric, FirstPickRateMetric,
RankStratifiedTakeRateMetric) share this exact accumulate() sequence -
look up the row's pick, iterate every pack_card_<name> column present,
tally in_pack/picked per key - and differ only in which rows are
eligible (_is_eligible_row()) and what key each tally is filed under
(_tally_key()). That's a Template Method (PATTERNS.md): this class owns
every shared step; a subclass only fixes KEY_COLUMNS/DEFAULT_OUTPUT_PATH
and implements _tally_key(), optionally overriding _is_eligible_row().
Mirrors sts_gg/card_average_metric.py's CardAverageMetric relationship
to its own nine subclasses.

pick_number_decay_curve_metric.py's PickNumberDecayCurveMetric also
subclasses this base (reusing accumulate()/_tally_key() unchanged) but
overrides finalize() entirely, since its output is one row per CARD (a
pick_number-indexed vector), not one row per key - see that module's
own docstring for why this is a deliberate, documented exception to
the shared finalize() below, rather than evidence this base is wrong.

UNMATCHED CARDS ARE NEVER SEEN HERE: unlike sts_gg's per-row CardBinder
lookup (which logs and excludes one unmatched deck entry per
accumulate() call), this container matches every card name once, up
front, via a DraftCardColumns (../pack_pool_columns.py) each metric
builds for itself in its own __init__ - an unmatched pack_card_<name>
column is simply absent from DraftCardColumns.pack_columns, so it can
never reach this class's accumulate() at all. See
DraftCardColumns.unmatched_names for where that bookkeeping actually
lives.

CARD_BINDER + HEADER, NOT A SHARED DraftCardColumns: each metric takes
(card_binder, header, source_game) directly - mirroring sts_gg's
CardAverageMetric/DeckLabelMetric always taking card_binder in their
own constructors - and builds its own DraftCardColumns internally
(DraftCardColumns.from_header()) rather than receiving an
already-built one from a shared driver. Re-parsing the same header once
per metric is a header-sized cost (hundreds of columns), not a
row-sized one, so it's cheap enough to duplicate per metric instead of
threading a second constructor argument through every metric just to
avoid it.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, Iterable
from uuid import UUID

import pandas as pd

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.metrics.seventeenlands.draft_data.pack_pool_columns import (
    DraftCardColumns,
)
from src.schema.game_id import GameId


class PackCardTallyMetric(ABC):
    """Per-(card, key) running (times_in_pack, times_picked) tally.

    Satisfies the Metric[dict] Protocol (../../metric.py) structurally.
    """

    KEY_COLUMNS: ClassVar[tuple[str, ...]]  # output column name(s) for
    # the non-card_uuid part of _tally_key()'s return - () for a
    # subclass whose key is the card alone (e.g. FirstPickRateMetric).
    DEFAULT_OUTPUT_PATH: ClassVar[Path]

    def __init__(
        self,
        card_binder: CardBinder,
        header: Iterable[str],
        source_game: GameId,
        output_path: Path | None = None,
    ) -> None:
        """
        Inputs:
            card_binder: registry to match this CSV's pack_card_<name>/
                pool_<name> column suffixes (and each row's pick cell
                value) against - assumed already fully populated for
                source_game. Never queried directly by this class -
                only through the DraftCardColumns this constructor
                builds from it.
            header: this CSV's column names (e.g.
                pandas.read_csv(path, nrows=0).columns) - parsed once,
                here, into this instance's own DraftCardColumns (see
                module docstring).
            source_game: which game's cards header/pick names are
                matched against.
            output_path: overrides DEFAULT_OUTPUT_PATH when given.
        Output: none (constructor).
        Side effects: none beyond building this instance's own
            DraftCardColumns from card_binder/header - no further I/O
            happens until finalize() is called.
        Exceptions: none.
        """
        self._draft_columns = DraftCardColumns.from_header(
            header, card_binder, source_game
        )
        self._output_path = output_path or self.DEFAULT_OUTPUT_PATH
        self._times_in_pack: dict[tuple, int] = {}
        self._times_picked: dict[tuple, int] = {}

    def accumulate(self, row: dict) -> None:
        """Tally every pack option this row's pack presents toward
        this metric's running per-key (times_in_pack, times_picked)
        counts.

        Inputs:
            row: one draft_data CSV row, dict-like - see
                ../scanner.py's module docstring.
        Output: none.
        Side effects: updates self._times_in_pack/_times_picked in
            place, once per matched pack_card_<name> column present on
            an eligible row.
        Exceptions: implementation-defined (expected: none for a
            well-formed row - see ../scanner.py's isolation contract
            for how a raised exception here is actually handled during
            a real scan).

        Example:
            >>> metric = SomeTallyMetric(card_binder, header, GameId.MTG)
            >>> metric.accumulate(row)
            >>> metric.finalize()
        """
        if not self._is_eligible_row(row):
            return

        pick_uuid = self._draft_columns.uuid_for_name(row["pick"])

        # Tally every matched, present pack option toward its own key.
        for card_uuid in self._draft_columns.present_uuids(
            row, self._draft_columns.pack_columns
        ):
            key = self._tally_key(row, card_uuid)
            self._times_in_pack[key] = self._times_in_pack.get(key, 0) + 1
            if card_uuid == pick_uuid:
                self._times_picked[key] = self._times_picked.get(key, 0) + 1

    def finalize(self) -> Path:
        """Compute every seen key's take rate and write one row per key
        to self._output_path.

        Inputs: none (uses accumulated state).
        Output: self._output_path.
        Side effects: creates self._output_path's parent directories if
            missing; writes self._output_path (a parquet file with
            columns nocab_uuid: str, *self.KEY_COLUMNS, take_rate:
            float, sample_count: int - one row per key seen at least
            once).
        Exceptions: whatever pandas.DataFrame.to_parquet raises.

        Example:
            >>> metric.finalize()
            PosixPath('data/metrics/seventeenlands/draft_data/some_take_rate.parquet')
        """
        result: list[dict] = [self._take_rate_row(key) for key in self._times_in_pack]

        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(result).to_parquet(self._output_path, index=False)
        return self._output_path

    def _is_eligible_row(self, row: dict) -> bool:
        """Whether this row is even a valid sample for this metric.

        Default: every row is eligible. FirstPickRateMetric
        (pack_card_tally_metrics.py) overrides this to restrict to
        pack_number == 0, pick_number == 0.

        Inputs:
            row: one draft_data CSV row, dict-like.
        Output: True if this row should be tallied at all.
        Side effects: none.
        Exceptions: none expected.
        """
        return True

    @abstractmethod
    def _tally_key(self, row: dict, card_uuid: UUID) -> tuple:
        """Build this metric's tally key for one (row, pack-option) pair.

        The main step a subclass overrides - together with
        KEY_COLUMNS, defines what this metric stratifies its take rate
        by (e.g. card alone, card x pack_number x pick_number, card x
        rank).

        Inputs:
            row: one draft_data CSV row, dict-like.
            card_uuid: a pack option's already-matched nocab_uuid.
        Output: a tuple whose first element is card_uuid and whose
            remaining elements correspond 1:1 with self.KEY_COLUMNS.
        Side effects: none expected.
        Exceptions: implementation-defined.
        """
        raise NotImplementedError

    def _take_rate_row(self, key: tuple) -> dict:
        """Build one output row for a single already-tallied key.

        Private helper - single consumer is finalize().

        Inputs:
            key: a key already present in self._times_in_pack.
        Output: a dict with keys "nocab_uuid" (str, key[0]), one entry
            per self.KEY_COLUMNS (from key[1:]), "take_rate" (float,
            self._times_picked.get(key, 0) / self._times_in_pack[key]),
            and "sample_count" (int, self._times_in_pack[key]).
        Side effects: none.
        Exceptions: none.
        """
        card_uuid, *stratifying_values = key
        times_in_pack = self._times_in_pack[key]
        times_picked = self._times_picked.get(key, 0)

        row: dict = {"nocab_uuid": str(card_uuid)}
        row.update(zip(self.KEY_COLUMNS, stratifying_values))
        row["take_rate"] = times_picked / times_in_pack
        row["sample_count"] = times_in_pack
        return row
