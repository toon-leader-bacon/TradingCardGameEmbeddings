"""v2 accumulator-shaped stress-test metric — see plans/metrics_v2.md.

Win rate = fraction of games where a card was in the deck that were
also won. A v2 port of src/data_refinement/seventeenlands/
game_data_metrics/metrics/win_rate/win_rate.py's WinRateMetric (via
base.py's _BinaryTriggerWinRateMetric) — the v1 version stays exactly
as it is, still in active use, not modified or replaced by this file
(see plans/metrics_v2.md's "Explicitly out of scope"). Deliberately
NOT ported alongside its two v1 siblings (DrawnWinRateMetric,
OpeningHandWinRateMetric) or the shared _BinaryTriggerWinRateMetric
base they extend — this stress test only needs ONE accumulator-shaped
example, and porting the whole win_rate/ family is exactly the kind of
per-metric migration plans/metrics_v2.md defers to later, one at a
time.

Chosen specifically so its output can be diffed against the
already-shipped v1 WinRateMetric's output on the same input CSV — the
strongest correctness check available for this stress test, since
there's a known-correct baseline to compare against.

accumulate()'s own win/games-counting logic (vectorized boolean masks,
summed per chunk) is unchanged from v1 — that part was never
checkpoint- or column-lookup-shaped to begin with. What changes: this
class looks up its own card columns via a held CardColumnCache
(card_column_cache.py, shared with deck_outcome_metric.py's identical
need) instead of receiving a scanner-supplied card_columns list, and
finalize() writes its own output file and returns its path instead of
returning a dict for a shared writer to merge.
"""

import json
from pathlib import Path
from typing import ClassVar
from uuid import UUID

import pandas as pd

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.seventeenlands.v2.game_data_metrics.card_column_cache import (
    CardColumnCache,
)
from src.schema.game_id import GameId

_METRIC_NAME = "win_rate"


class WinRateMetric:
    """Accumulator Metric: running per-card wins/games counts, written
    out once, in finalize().

    Satisfies v2's Metric Protocol (src/data_refinement/seventeenlands/
    v2/metric.py) structurally.
    """

    name: ClassVar[str] = _METRIC_NAME
    DEFAULT_OUTPUT_DIR: ClassVar[Path] = Path("data/final/metrics/17lands/v2/game")
    DEFAULT_OUTPUT_NAME: ClassVar[str] = "{expansion}.{format_code}.win_rate.jsonl"
    # Same v2 output namespace as DeckOutcomeMetric, distinct from v1's
    # MetricScanner-written parquet at data/final/metrics/17lands/game/
    # — the two are not guaranteed to produce byte-identical files, so
    # sharing a path would risk one overwriting the other. JSONL here
    # too (not parquet), for consistency with the rest of v2's output
    # and to keep this stress test's own writer trivial — one JSON
    # object per resolved card, same fields v1's MetricResult carries
    # (nocab_uuid, metric_name, value, sample_size, expansion, format).

    _card_columns: CardColumnCache  # constructed in __init__

    def __init__(
        self,
        card_binder: CardBinder,
        source_game: GameId,
        expansion: str,
        format_code: str,
        output_path: Path | None = None,
    ) -> None:
        """
        Inputs:
            card_binder: registry this metric resolves its own
                deck_<name> columns against.
            source_game: which game card_binder's aliases should be
                resolved against.
            expansion: 17lands expansion code, stamped onto every
                output row.
            format_code: 17lands format code, stamped onto every
                output row.
            output_path: where this metric's own JSONL output is
                written. Defaults to
                WinRateMetric.default_output_path(expansion,
                format_code) when not given.
        Output: none (constructor).
        Side effects: none — no I/O happens until accumulate() is
            first called.
        Exceptions: none.
        """
        self._expansion = expansion
        self._format_code = format_code
        self._output_path = output_path or self.default_output_path(expansion, format_code)
        self._card_columns = CardColumnCache(card_binder, source_game)
        self._wins: dict[UUID, int] = {}
        self._games: dict[UUID, int] = {}

    @staticmethod
    def default_output_path(expansion: str, format_code: str) -> Path:
        """The conventional output_path for one (expansion, format_code) run.

        Inputs:
            expansion: 17lands expansion code.
            format_code: 17lands format code.
        Output: DEFAULT_OUTPUT_DIR / DEFAULT_OUTPUT_NAME, formatted
            with expansion/format_code.
        Side effects: none — purely a path computation.
        Exceptions: none.

        Example:
            >>> WinRateMetric.default_output_path("MSH", "PremierDraft")
            PosixPath('data/final/metrics/17lands/v2/game/MSH.PremierDraft.win_rate.jsonl')
        """
        return WinRateMetric.DEFAULT_OUTPUT_DIR / WinRateMetric.DEFAULT_OUTPUT_NAME.format(
            expansion=expansion, format_code=format_code
        )

    def accumulate(self, chunk: pd.DataFrame) -> None:
        """Update wins/games counts for every known card, from one chunk.

        Composed of: self._card_columns.card_columns(chunk) (a
        CardColumnCache — lazily looks up card columns from chunk's
        own header, once, on the first call — see this module's own
        docstring), then the unchanged-from-v1 counting logic: for
        each CardColumnSet, a game counts toward that card's
        denominator if chunk[card_column_set.deck] > 0, and
        additionally toward the numerator if chunk["won"] is also True
        for that row — both computed via vectorized pandas boolean
        masks summed over the whole chunk, matching
        game_data_metrics/metrics/win_rate/base.py's
        _BinaryTriggerWinRateMetric.accumulate() exactly.

        Inputs:
            chunk: one chunk of a raw game_data CSV — must include a
                "won" column and every card's deck_<name> column.
        Output: none.
        Side effects: on the first call, looks up card columns;
            increments self._wins and self._games in place (adds this
            chunk's counts to whatever was already accumulated — never
            overwrites/resets either dict). A card with zero observed
            games in this chunk is left entirely untouched.
        Exceptions: none expected from well-formed input.
        """
        card_columns = self._card_columns.card_columns(chunk)
        for card_column_set in card_columns:
            triggered = chunk[card_column_set.deck] > 0
            games_this_chunk = int(triggered.sum())
            if games_this_chunk == 0:
                continue
            wins_this_chunk = int((triggered & chunk["won"]).sum())

            uuid = card_column_set.nocab_uuid
            self._games[uuid] = self._games.get(uuid, 0) + games_this_chunk
            self._wins[uuid] = self._wins.get(uuid, 0) + wins_this_chunk

    def finalize(self) -> Path:
        """Turn accumulated wins/games counts into JSON lines, write
        them to self._output_path, and report that path.

        One JSON object per nocab_uuid present in self._games:
        {"nocab_uuid": <uuid str>, "metric_name": self.name, "value":
        self._wins[uuid] / games, "sample_size": games, "expansion":
        self._expansion, "format": self._format_code}. No
        minimum-sample-size cutoff — every card with at least one
        observed game gets a result, matching v1's WinRateMetric
        exactly.

        Inputs: none (uses self._wins, self._games).
        Output: self._output_path.
        Side effects: creates self._output_path's parent directory if
            missing; writes self._output_path (even if empty — zero
            resolved cards still produces a valid, zero-line file).
        Exceptions: raises on failure to write self._output_path.

        Example:
            >>> metric = WinRateMetric(card_binder, GameId.MTG, "MSH", "PremierDraft")
            >>> # ... scanner drives accumulate() per chunk ...
            >>> output_path = metric.finalize()
        """
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._output_path, "w") as output_file:
            for uuid, games in self._games.items():
                row = {
                    "nocab_uuid": str(uuid),
                    "metric_name": self.name,
                    "value": self._wins[uuid] / games,
                    "sample_size": games,
                    "expansion": self._expansion,
                    "format": self._format_code,
                }
                output_file.write(json.dumps(row) + "\n")
        return self._output_path
