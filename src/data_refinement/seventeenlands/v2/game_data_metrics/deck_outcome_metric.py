"""v2 streaming-shaped stress-test metric — see plans/metrics_v2.md.

Reconstructs one row's deck (with duplicates — a card run as a 2-of
appears twice) plus that game's won outcome, appends it as one JSON
line to this metric's own output file. Unlike
src/data_refinement/seventeenlands/game_data_metrics/
deck_outcome_scanner.py (the standalone v1-era class this is modeled
on, still in active use, not modified or replaced by this file — see
plans/metrics_v2.md's "Explicitly out of scope"), this class:
- looks up its own card columns, lazily, from the first chunk's own
  header, via a held CardColumnCache (card_column_cache.py, shared
  with win_rate_metric.py's identical need) — no raw_csv_path, no
  separate header-read step; the chunk it's already being handed
  already contains the header information it needs.
- has no checkpoint/resume machinery at all — a failed scan is simply
  rerun from row 0 (plans/metrics_v2.md's explicit, deliberate
  simplification).
- writes JSONL not for crash-durability (not a requirement here) but
  because deck_card_uuids is list-valued per row, which JSONL
  represents natively.
"""

import json
from pathlib import Path
from typing import IO, ClassVar

import pandas as pd

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.seventeenlands.game_data_metrics.card_column_set import (
    CardColumnSet,
)
from src.data_refinement.seventeenlands.v2.game_data_metrics.card_column_cache import (
    CardColumnCache,
)
from src.schema.game_id import GameId

_METRIC_NAME = "deck_outcome"


class DeckOutcomeMetric:
    """Streaming Metric: one JSON line per raw CSV row, written as
    each row is seen.

    Satisfies v2's Metric Protocol (src/data_refinement/seventeenlands/
    v2/metric.py) structurally.
    """

    name: ClassVar[str] = _METRIC_NAME
    DEFAULT_OUTPUT_DIR: ClassVar[Path] = Path("data/final/metrics/17lands/v2/game")
    DEFAULT_OUTPUT_NAME: ClassVar[str] = "{expansion}.{format_code}.deck_outcomes.jsonl"
    # A distinct output namespace (data/final/metrics/17lands/v2/...,
    # not .../17lands/game/...) from v1's DeckOutcomeScanner — the two
    # are not guaranteed to produce byte-identical output, so sharing
    # a path would risk one silently overwriting the other.

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
                DeckOutcomeMetric.default_output_path(expansion,
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
        self._output_file: IO[str] | None = None

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
            >>> DeckOutcomeMetric.default_output_path("MSH", "PremierDraft")
            PosixPath('data/final/metrics/17lands/v2/game/MSH.PremierDraft.deck_outcomes.jsonl')
        """
        return DeckOutcomeMetric.DEFAULT_OUTPUT_DIR / DeckOutcomeMetric.DEFAULT_OUTPUT_NAME.format(
            expansion=expansion, format_code=format_code
        )

    def accumulate(self, chunk: pd.DataFrame) -> None:
        """Reconstruct and write one JSON line per row of chunk.

        Composed of: self._card_columns.card_columns(chunk) (a
        CardColumnCache — see card_column_cache.py — lazily looks up
        card columns from chunk's own header on the first call, a
        no-op cache read on every later call), _ensure_output_file_open()
        (lazily opens self._output_path in write mode on the first
        call), then for every row: builds a DeckOutcome-shaped record
        via _build_deck_outcome() and writes it via
        _write_deck_outcome_line().

        Inputs:
            chunk: one chunk of a raw game_data CSV — must include a
                "won" column and every card's deck_<name> column.
        Output: none.
        Side effects: on the first call, looks up card columns and
            opens self._output_path for writing; every call appends
            len(chunk) JSON lines to that file.
        Exceptions: raises on failure to write self._output_path.
        """
        card_columns = self._card_columns.card_columns(chunk)
        output_file = self._ensure_output_file_open()
        for _, row in chunk.iterrows():
            deck_outcome = self._build_deck_outcome(row, card_columns)
            self._write_deck_outcome_line(output_file, deck_outcome)

    def finalize(self) -> Path:
        """Close this metric's output file (opening an empty one first
        if accumulate() was never called) and report its path.

        Inputs: none.
        Output: self._output_path.
        Side effects: closes this metric's output file handle if open;
            creates an empty (header-less, zero-line) output file if
            accumulate() was never called, so this metric's output
            path always exists after finalize(), even for an empty
            input.
        Exceptions: raises on failure to write/close self._output_path.

        Example:
            >>> metric = DeckOutcomeMetric(card_binder, GameId.MTG, "MSH", "PremierDraft")
            >>> # ... scanner drives accumulate() per chunk ...
            >>> output_path = metric.finalize()
        """
        output_file = self._ensure_output_file_open()
        output_file.close()
        return self._output_path

    def _ensure_output_file_open(self) -> IO[str]:
        """Open self._output_path for writing, once, caching the handle.

        Private helper — single consumer is accumulate()/finalize().
        Creates self._output_path's parent directory if missing.

        Inputs: none (uses self._output_path).
        Output: the open, writable file handle (cached — every call
            after the first returns the same handle).
        Side effects: on the first call, creates self._output_path's
            parent directory and opens self._output_path in write
            mode; a no-op on later calls.
        Exceptions: raises on failure to open self._output_path.
        """
        if self._output_file is None:
            self._output_path.parent.mkdir(parents=True, exist_ok=True)
            self._output_file = open(self._output_path, "w")
        return self._output_file

    def _build_deck_outcome(self, row: pd.Series, card_columns: list[CardColumnSet]) -> dict:
        """Reconstruct one game row's deck + outcome as a JSON-ready dict.

        Private helper — single consumer is accumulate(). For every
        CardColumnSet in card_columns, reads that card's deck_<name>
        cell as an integer count; if the count is > 0, that card's
        nocab_uuid (stringified — JSON has no native UUID type)
        appears in the returned deck_card_uuids exactly that many
        times (duplicates included). A card whose column wasn't found
        in card_columns simply never contributes an entry — this never
        causes the row itself to be skipped.

        Inputs:
            row: one row of a chunk passed to accumulate() — must
                include every card_columns entry's .deck column, plus
                "won".
            card_columns: this scan's card columns, from
                self._card_columns.card_columns(chunk).
        Output: {"deck_card_uuids": [<uuid str>, ...], "won": <bool>,
            "expansion": self._expansion, "format": self._format_code}.
        Side effects: none.
        Exceptions: none expected from well-formed input.
        """
        deck_card_uuids: list[str] = []
        for card_column_set in card_columns:
            count = int(row[card_column_set.deck])
            if count > 0:
                deck_card_uuids.extend([str(card_column_set.nocab_uuid)] * count)
        return {
            "deck_card_uuids": deck_card_uuids,
            "won": bool(row["won"]),
            "expansion": self._expansion,
            "format": self._format_code,
        }

    def _write_deck_outcome_line(self, output_file: IO[str], deck_outcome: dict) -> None:
        """Append one deck_outcome dict to output_file as a single JSON line.

        Private helper — single consumer is accumulate().

        Inputs:
            output_file: an already-open, writable file handle.
            deck_outcome: one _build_deck_outcome() result.
        Output: none.
        Side effects: writes one line (a JSON object plus a trailing
            newline) to output_file.
        Exceptions: raises on failure to write output_file.
        """
        output_file.write(json.dumps(deck_outcome) + "\n")
