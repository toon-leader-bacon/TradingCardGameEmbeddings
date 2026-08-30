"""Drives a checkpoint-free, resumable single pass over one 17lands
game_data CSV, emitting one DeckOutcome training example per row.

See plans/deck_outcome_dojo.md's Component 1 for the full design this
codifies. Deliberately does NOT implement ScannableMetric/Metric and
does NOT compose accumulate_over_chunks()/write_metric_results()/
MetricCheckpoint (metric_scan_loop.py, metric_writer.py,
metric_checkpoint.py) — those are all shaped around "accumulate per
chunk, finalize once into a final aggregate," which fits every
Metric's own job but not this one: this class emits one independent
example per raw CSV row, with no aggregate state to accumulate at all.

Output is JSONL, not parquet — every other artifact in this project is
parquet, but a parquet file only becomes valid/readable once its
footer is written at close, so it cannot be safely resumed from a
crash mid-write the way this class needs. JSONL's per-line durability
(a complete line stays parseable even if the process dies mid-write on
the next one) is what makes this class's own output file the sole,
resume-authoritative source of truth — there is no separate checkpoint
file, and no checkpoint_path/checkpoint_every_n_chunks constructor
parameter exists on this class.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import IO, ClassVar
from uuid import UUID

import pandas as pd

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.seventeenlands.game_data_metrics.card_column_set import (
    CardColumnSet,
)
from src.data_refinement.seventeenlands.game_data_metrics.column_lookup import (
    ColumnResolution,
    find_card_columns_in_csv,
)
from src.data_refinement.seventeenlands.game_data_metrics.metric_scanner import (
    MetricScanner,
)
from src.schema.game_id import GameId

_CHUNK_SIZE_ROWS = 100_000
# Not shared with metric_scan_loop.py's own _CHUNK_SIZE_ROWS constant —
# this class deliberately doesn't compose that module's
# accumulate_over_chunks() loop (see this module's own docstring), so
# there's no shared call site to centralize the constant through
# without pulling in the accumulate-then-finalize machinery this class
# is specifically avoiding.


@dataclass(frozen=True)
class DeckOutcome:
    """One reconstructed deck + the outcome of the single game it was
    played in.

    In-memory/dataclass shape only — see this module's own docstring
    and _write_deck_outcome_line()'s for the str-per-uuid JSONL
    round-trip this type's deck_card_uuids field goes through on disk.

    Inputs: none (data holder).
    Output: n/a.
    Side effects: none.
    Exceptions: none.
    """

    deck_card_uuids: list[UUID]  # duplicates included — a card run as
    # a 2-of in this game's tracked maindeck appears twice here.
    won: bool
    expansion: str
    format: str


@dataclass(frozen=True)
class DeckOutcomeScanResult:
    """What one DeckOutcomeScanner.scan() call produced.

    Inputs: none (data holder).
    Output: n/a.
    Side effects: none.
    Exceptions: none.
    """

    output_path: Path
    unresolved_column_names: list[str]  # card names in the CSV header
    # CardBinder couldn't resolve — same meaning as MetricScanResult's
    # own field.


class DeckOutcomeScanner:
    """Drives a resumable single pass over one game_data CSV, writing
    one DeckOutcome JSON line per row.

    Single-consumer-per-scan — one DeckOutcomeScanner instance handles
    one (raw_csv_path, output_path) combination; a different CSV or a
    different output location gets its own instance. Lives alongside
    MetricScanner (this same directory) since both read the same raw
    game_data CSV shape via the same column-resolution logic, but
    produces a structurally different output artifact (see this
    module's own docstring) — not a MetricScanner subclass or
    variant.
    """

    DEFAULT_OUTPUT_DIR: ClassVar[Path] = MetricScanner.DEFAULT_OUTPUT_DIR
    # Same data/final/metrics/17lands/game/ directory MetricScanner
    # itself writes into.
    DEFAULT_OUTPUT_NAME: ClassVar[str] = "{expansion}.{format_code}.deck_outcomes.jsonl"
    # Distinct filename AND extension from MetricScanner's own
    # DEFAULT_OUTPUT_NAME — the two coexist in the same directory
    # without colliding.

    def __init__(
        self,
        raw_csv_path: Path,
        card_binder: CardBinder,
        output_path: Path,
        source_game: GameId,
        expansion: str,
        format_code: str,
    ) -> None:
        """
        Inputs:
            raw_csv_path: path to a 17lands game_data CSV (e.g.
                data/raw/17lands/game_data/MSH.PremierDraft.csv).
            card_binder: registry to resolve deck_<name> columns
                against.
            output_path: where DeckOutcome JSON lines are
                written/appended. If this path already contains
                output from a prior, interrupted scan, scan() resumes
                from it rather than starting over — see scan()'s own
                docstring.
            source_game: which game raw_csv_path's cards belong to —
                same reason MetricScanner takes this (CardBinder is
                generic across GameId).
            expansion: 17lands expansion code, stamped onto every
                DeckOutcome this scan produces.
            format_code: 17lands format code, stamped onto every
                DeckOutcome this scan produces (as DeckOutcome.format).
        Output: none (constructor).
        Side effects: none — no I/O happens until scan() is called.
        Exceptions: none.
        """
        self.raw_csv_path = raw_csv_path
        self.card_binder = card_binder
        self.output_path = output_path
        self.source_game = source_game
        self.expansion = expansion
        self.format_code = format_code

    @staticmethod
    def default_output_path(expansion: str, format_code: str) -> Path:
        """The conventional output_path for one (expansion, format_code) scan.

        A recommended default, not an enforced requirement —
        output_path stays a required constructor parameter regardless,
        same convention as every other *Scanner in this project.

        Inputs:
            expansion: 17lands expansion code.
            format_code: 17lands format code.
        Output: DEFAULT_OUTPUT_DIR / DEFAULT_OUTPUT_NAME, formatted
            with expansion/format_code (e.g.
            data/final/metrics/17lands/game/MSH.PremierDraft.deck_outcomes.jsonl).
        Side effects: none — purely a path computation.
        Exceptions: none.

        Example:
            >>> DeckOutcomeScanner.default_output_path("MSH", "PremierDraft")
            PosixPath('data/final/metrics/17lands/game/MSH.PremierDraft.deck_outcomes.jsonl')
        """
        return (
            DeckOutcomeScanner.DEFAULT_OUTPUT_DIR
            / DeckOutcomeScanner.DEFAULT_OUTPUT_NAME.format(
                expansion=expansion, format_code=format_code
            )
        )

    def scan(self) -> DeckOutcomeScanResult:
        """Resolve columns, determine resume state, stream the CSV,
        and write DeckOutcome JSON lines.

        Composed of:
            1. _find_card_columns() — resolves raw_csv_path's header
               against card_binder (mirrors MetricScanner's own
               private helper of the same name/shape).
            2. _prepare_output_file() — determines how many rows were
               already durably processed by a prior, interrupted scan
               (0 for a fresh scan), truncating away any trailing
               partial JSON line left by a crash mid-write. See that
               method's own docstring for the exact algorithm this
               plan already locked in.
            3. _stream_and_write() — streams raw_csv_path in chunks
               starting from the resume row offset, building and
               appending one DeckOutcome JSON line per row, flushing
               once per chunk.
            4. Constructs and returns the DeckOutcomeScanResult.

        Inputs: none (uses constructor-supplied state).
        Output: a DeckOutcomeScanResult naming output_path and every
            card name from raw_csv_path's header that couldn't be
            resolved.
        Side effects: reads raw_csv_path and (if present) output_path;
            truncates output_path if it contains a trailing partial
            line from a prior crash; appends to output_path throughout
            the scan.
        Exceptions: raises on failure to read raw_csv_path, or on
            failure to read/write/truncate output_path.

        Example:
            >>> scanner = DeckOutcomeScanner(
            ...     Path("data/raw/17lands/game_data/MSH.PremierDraft.csv"),
            ...     card_binder,
            ...     Path("data/final/metrics/17lands/game/MSH.PremierDraft.deck_outcomes.jsonl"),
            ...     GameId.MTG,
            ...     "MSH",
            ...     "PremierDraft",
            ... )
            >>> result = scanner.scan()
        """
        column_resolution = self._find_card_columns()
        rows_already_processed = self._prepare_output_file()
        self._stream_and_write(column_resolution.resolved, rows_already_processed)
        return DeckOutcomeScanResult(
            output_path=self.output_path,
            unresolved_column_names=column_resolution.unresolved_names,
        )

    def _find_card_columns(self) -> ColumnResolution:
        """Resolve raw_csv_path's header against card_binder.

        Private helper — single consumer is scan(). Delegates to
        column_lookup.find_card_columns_in_csv(), shared with
        MetricScanner's own identical header-read-then-resolve need
        (metric_scanner.py) — both scanners resolve the same
        game_data CSV column shape, just for different downstream
        purposes.

        Inputs: none (uses self.raw_csv_path, self.card_binder,
            self.source_game).
        Output: a ColumnResolution splitting the header's card names
            into resolved CardColumnSets and unresolved names.
        Side effects: reads raw_csv_path's header row.
        Exceptions: raises if raw_csv_path doesn't exist or has no
            readable header row.
        """
        return find_card_columns_in_csv(self.raw_csv_path, self.card_binder, self.source_game)

    def _prepare_output_file(self) -> int:
        """Determine the resume row offset and truncate output_path to
        remove any trailing partial line from a prior crash.

        Private helper — single consumer is scan(). If output_path
        doesn't exist yet, this is a fresh scan: returns 0, performs
        no truncation. If output_path already exists, reads it line by
        line, parsing each as JSON, tracking two separate values as
        lines are confirmed valid: a count of valid lines (this
        method's return value) and the file's byte position
        immediately after the last successfully parsed line (via
        file.tell(), taken right after that line — NOT the line count
        itself, a different unit). Reading stops at, and discards, the
        first line that fails to parse (a trailing partial line from a
        crash mid-write). The file is then truncated to that tracked
        byte position, leaving it ready to reopen in append mode with
        no partial trailing line mixed in with newly appended rows.

        Inputs: none (uses self.output_path).
        Output: the number of rows already durably processed by a
            prior scan (0 for a fresh scan) — the value scan() feeds
            to pandas.read_csv's skiprows when resuming.
        Side effects: truncates output_path if it exists and its last
            line is incomplete; no effect if output_path doesn't exist
            or its last line already parses cleanly.
        Exceptions: raises on failure to read or truncate output_path.
        """
        if not self.output_path.exists():
            return 0

        valid_line_count = 0
        safe_byte_offset = 0
        with self.output_path.open("r") as existing_file:
            # readline() in an explicit loop, not `for line in
            # existing_file` — mixing that iteration form with tell()
            # raises "OSError: telling position disabled by next()
            # call" (CPython's text-mode iterator uses an internal
            # read-ahead buffer that makes tell() unreliable); readline()
            # doesn't use that buffer, so tell() stays accurate.
            while True:
                line = existing_file.readline()
                if not line:
                    break
                try:
                    json.loads(line)
                except json.JSONDecodeError:
                    break
                valid_line_count += 1
                safe_byte_offset = existing_file.tell()

        with self.output_path.open("r+b") as truncate_file:
            truncate_file.truncate(safe_byte_offset)

        return valid_line_count

    def _stream_and_write(
        self, resolved_card_columns: list[CardColumnSet], rows_already_processed: int
    ) -> None:
        """Stream raw_csv_path in chunks, appending one DeckOutcome
        JSON line per row to output_path.

        Private helper — single consumer is scan(). Opens output_path
        in append mode once for the whole call (not reopened per
        chunk). Streams raw_csv_path via
        pandas.read_csv(chunksize=_CHUNK_SIZE_ROWS, skiprows=...),
        skipping rows_already_processed rows when resuming. For each
        chunk: builds a DeckOutcome per row via _build_deck_outcome()
        and writes it via _write_deck_outcome_line(), then flushes the
        file once the whole chunk's rows are written.

        Inputs:
            resolved_card_columns: every resolved card's column names
                for this scan, from _find_card_columns() — unchanged
                across every chunk (the header is resolved once, not
                per chunk).
            rows_already_processed: how many leading data rows to skip
                (0 for a fresh scan, _prepare_output_file()'s return
                value when resuming).
        Output: none.
        Side effects: reads raw_csv_path; appends to and flushes
            output_path repeatedly throughout the call.
        Exceptions: raises on failure to read raw_csv_path or write
            output_path.
        """
        skiprows = range(1, rows_already_processed + 1) if rows_already_processed > 0 else None
        chunk_iterator = pd.read_csv(
            self.raw_csv_path, chunksize=_CHUNK_SIZE_ROWS, skiprows=skiprows
        )

        with self.output_path.open("a") as output_file:
            for chunk in chunk_iterator:
                for _, row in chunk.iterrows():
                    deck_outcome = self._build_deck_outcome(row, resolved_card_columns)
                    self._write_deck_outcome_line(output_file, deck_outcome)
                output_file.flush()

    def _build_deck_outcome(
        self, row: pd.Series, resolved_card_columns: list[CardColumnSet]
    ) -> DeckOutcome:
        """Reconstruct one game row's deck + outcome as a DeckOutcome.

        Private helper — single consumer is _stream_and_write(). Pure
        transform, no side effects. For every CardColumnSet in
        resolved_card_columns, reads that card's deck_<name> cell as
        an integer count; if the count is > 0, that card's nocab_uuid
        appears in the returned deck_card_uuids exactly that many
        times (duplicates included — see DeckOutcome's own docstring).
        A card whose column is unresolved (not present in
        resolved_card_columns at all — see _find_card_columns())
        simply never contributes an entry; this never causes the row
        itself to be skipped.

        Inputs:
            row: one row of a raw_csv_path chunk — must include every
                resolved_card_columns entry's .deck column, plus
                "won".
            resolved_card_columns: every resolved card's column names
                for this scan, unchanged across calls.
        Output: one DeckOutcome, stamped with self.expansion/
            self.format_code.
        Side effects: none.
        Exceptions: none expected from well-formed input.
        """
        deck_card_uuids: list[UUID] = []
        for card_column_set in resolved_card_columns:
            count = int(row[card_column_set.deck])
            if count > 0:
                deck_card_uuids.extend([card_column_set.nocab_uuid] * count)
        return DeckOutcome(
            deck_card_uuids=deck_card_uuids,
            won=bool(row["won"]),
            expansion=self.expansion,
            format=self.format_code,
        )

    def _write_deck_outcome_line(self, output_file: IO[str], deck_outcome: DeckOutcome) -> None:
        """Append one DeckOutcome to output_file as a single JSON line.

        Private helper — single consumer is _stream_and_write().
        deck_card_uuids is written as a list of strings (str(uuid) per
        entry — JSON has no native UUID type), following the same
        string round-trip convention metric_writer.py/
        MetricRegressionDojo.prepare_splits already use for parquet
        output elsewhere in this project. Does NOT flush — flushing
        happens once per chunk in _stream_and_write(), not once per
        row.

        Inputs:
            output_file: an already-open, writable file handle
                (append mode), positioned at the end.
            deck_outcome: the row to write.
        Output: none.
        Side effects: writes one line (a JSON object plus a trailing
            newline) to output_file.
        Exceptions: raises on failure to write output_file.
        """
        line = json.dumps(
            {
                "deck_card_uuids": [str(uuid) for uuid in deck_outcome.deck_card_uuids],
                "won": deck_outcome.won,
                "expansion": deck_outcome.expansion,
                "format": deck_outcome.format,
            }
        )
        output_file.write(line + "\n")
