"""Converts sts_gg run data into a per-run deck/outcome metric.

TECHNOLOGY DEMONSTRATION, NOT A CONSIDERED TRAINING SIGNAL: this is the
first metric in this project built from two independent raw data
sources for the same game — spire_codex supplies Slay the Spire 2's
card identities (via its own CardIngestionStage, already ingested into
the CardBinder), sts_gg supplies run data referencing those same cards
under a different id scheme. The whole point of this file is to show
that a second source's card references resolve through the *same*
CardBinder a different source's ingestion stage built, not to produce
a metric worth training on. It's a weak signal on its own merits: a
model could plausibly infer a lot of the win/loss label just from
whether starting Strikes/Defends are still in the deck (a real player
tends to prune those early in a winning run) — a stronger version of
this idea (e.g. P(wins | survived to floor Y)) is future work, not
this file's concern.

CARD RESOLUTION: sts_gg's raw deck entries carry ids like
"CARD.GRAND_FINALE"; spire_codex/cards.json's own "id" field for the
same card is "GRAND_FINALE" — no prefix. Confirmed live (2026-09-05,
by direct inspection of data/raw/sts_gg/runs.jsonl and
data/raw/spire_codex/cards.json) that stripping the literal "CARD."
prefix reproduces spire_codex's own id exactly. This is NOT a fuzzy/
name-based match like src/data_refinement/seventeenlands/
name_lookup.py's find_uuid_by_name (which exists only because 17lands
gives MTG card names as plain text with no shared id at all) — it's a
deterministic string transform followed by a plain alias lookup
against spire_codex's own already-registered
CardBinder.get_by_alias(GameId.SLAY_THE_SPIRE_2, DataSource.SPIRE_CODEX,
...) aliases. Deliberately no new DataSource enum value and no
ingestion stage for sts_gg: it doesn't mint its own card identity, it
just borrows spire_codex's under a different string prefix.

UNRESOLVED CARDS: if a deck's card id doesn't resolve (e.g.
spire_codex's dump lagging a newer sts.gg game build), it's logged
loudly via the stdlib logging module (the first use of `logging`
anywhere in this codebase — data_retrieval's sibling containers use
tqdm.write for a different, progress-bar-safe purpose; data_refinement
has no prior convention) and excluded from just that one run's
deck_nocab_uuids — the run itself is still emitted. Explicit design
choice: more (but occasionally dirty) training data beats less (but
perfectly clean) training data; a run is never dropped over one bad
card id.

STREAMING, NOT ACCUMULATION: each raw run produces one output row
independently of every other run — no cross-row aggregation is ever
needed (contrast AveragePickNumberMetric in
src/data_refinement/seventeenlands/draft_game_metrics/
average_pick_number_metric.py, an accumulation metric whose
finalize() does the real per-card averaging that accumulate() alone
can't). accumulate() here writes its row immediately via an already-
open pyarrow.parquet.ParquetWriter; finalize() only closes that
writer — a true no-op relative to *data*, unlike an accumulation
metric's finalize(). This is NOT the seventeenlands Metric Protocol
(src/data_refinement/seventeenlands/metric.py) — that protocol is
explicitly CsvScanner-driven, taking pd.DataFrame chunks; this
container's raw data is JSONL, one dict per line, with no equivalent
scanner today. This class doesn't implement that Protocol and doesn't
need to.
"""

import json
import logging
from pathlib import Path
from typing import ClassVar
from uuid import UUID

import pyarrow as pa
import pyarrow.parquet as pq

from src.data_refinement.card_binder.card_binder import CardBinder
from src.schema.data_source import DataSource
from src.schema.game_id import GameId

_logger = logging.getLogger(__name__)

_STS_GG_CARD_ID_PREFIX = "CARD."

# Fixed and known upfront — never inferred from data, unlike a
# CsvScanner-driven Metric's pandas dtypes.
_OUTPUT_SCHEMA = pa.schema(
    [
        ("run_id", pa.string()),
        ("deck_nocab_uuids", pa.list_(pa.string())),
        ("win", pa.bool_()),
    ]
)


class DeckOutcomeMetric:
    """Streaming per-run metric: one sts_gg run in, one
    (run_id, deck_nocab_uuids, win) row out, written immediately.

    Single-consumer to src/data_refinement/sts_gg/ — no other
    container depends on this class.
    """

    DEFAULT_OUTPUT_PATH: ClassVar[Path] = Path(
        "data/final/metrics/sts_gg/deck_outcome.parquet"
    )

    def __init__(self, card_binder: CardBinder, output_path: Path | None = None) -> None:
        """
        Inputs:
            card_binder: registry to resolve sts_gg's "CARD.<id>"
                references against — must already have spire_codex's
                cards ingested (this class never writes to it).
            output_path: overrides DEFAULT_OUTPUT_PATH when given.
        Output: none (constructor).
        Side effects: creates output_path's parent directories if
            missing; opens output_path for writing (truncating any
            existing file at that path) via a pyarrow.parquet.
            ParquetWriter held open for the lifetime of this instance —
            callers MUST call finalize() when done, or the file is left
            incomplete/unreadable. See scan_runs_jsonl for this
            project's one call site, which guarantees finalize() runs
            (via try/finally) even if a row fails partway through a
            scan.
        Exceptions: whatever pyarrow.parquet.ParquetWriter raises on
            failure to open output_path for writing.
        """
        self._card_binder = card_binder
        self._output_path = output_path or self.DEFAULT_OUTPUT_PATH
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = pq.ParquetWriter(self._output_path, _OUTPUT_SCHEMA)
        self._closed = False

    def accumulate(self, row: dict) -> None:
        """Convert one parsed sts_gg run into a single output row and
        write it immediately.

        Streaming, not accumulating (see module docstring): this row's
        output depends on nothing else this instance has seen or will
        see, so there's no reason to hold it in memory past this call.

        Inputs:
            row: one parsed JSON object from sts_gg's runs.jsonl (one
                run), carrying at least "id" (str), "win" (bool), and
                "deck" (list of {"id": str, ...} card entries — other
                per-entry fields like "upgraded"/"floor" describe a
                specific played instance, not card identity, and are
                not part of this metric's output).
        Output: none.
        Side effects: writes exactly one row to the open
            ParquetWriter — visible in output_path once finalize() has
            flushed/closed the writer, not necessarily before. Emits
            one logging.error() per card in this run that fails to
            resolve (see module docstring's UNRESOLVED CARDS policy).
        Exceptions: raises if row is missing "id", "win", or "deck".

        Example:
            >>> metric = DeckOutcomeMetric(card_binder)
            >>> metric.accumulate(json.loads(next(runs_file)))
            >>> metric.finalize()
        """
        run_id = row["id"]
        win = row["win"]

        # Resolve every deck entry's card id to a nocab_uuid, dropping
        # (and loudly logging) any that don't resolve — never dropping
        # the whole run over one bad card id.
        deck_nocab_uuids: list[str] = []
        for card_entry in row["deck"]:
            card_uuid = self._card_uuid(card_entry["id"])
            if card_uuid is None:
                _logger.error(
                    "DeckOutcomeMetric: run %s references unresolved card id %r "
                    "— excluding it from this run's deck_nocab_uuids",
                    run_id,
                    card_entry["id"],
                )
                continue
            deck_nocab_uuids.append(str(card_uuid))

        # Build this run's one-row Table against _OUTPUT_SCHEMA and
        # write it immediately.
        output_row = pa.Table.from_pydict(
            {
                "run_id": [run_id],
                "deck_nocab_uuids": [deck_nocab_uuids],
                "win": [win],
            },
            schema=_OUTPUT_SCHEMA,
        )
        self._writer.write_table(output_row)

    def finalize(self) -> Path:
        """Close the underlying ParquetWriter.

        A true no-op relative to data — every row this instance will
        ever write was already written by accumulate() (see module
        docstring). This only flushes/closes the file handle.

        Idempotent: a second call is a no-op rather than an error — see
        scan_runs_jsonl's try/finally, which calls this unconditionally
        even after an accumulate() failure, so a caller that also calls
        finalize() itself (per this method's own contract) shouldn't be
        punished for a writer that's already closed.

        Inputs: none.
        Output: output_path.
        Side effects: closes the ParquetWriter opened in __init__, if
            not already closed.
        Exceptions: whatever ParquetWriter.close() raises.

        Example:
            >>> metric.finalize()
            PosixPath('data/final/metrics/sts_gg/deck_outcome.parquet')
        """
        if not self._closed:
            self._writer.close()
            self._closed = True
        return self._output_path

    def _card_uuid(self, raw_card_id: str) -> UUID | None:
        """Look up the nocab_uuid for one sts_gg deck entry's card id.

        Private helper — single consumer is accumulate(). Strips
        sts_gg's "CARD." prefix and looks the remainder up directly
        against spire_codex's own registered aliases (see module
        docstring's CARD RESOLUTION section) — a plain alias lookup,
        not a fuzzy/name-based match.

        Inputs:
            raw_card_id: one deck entry's "id" field, e.g.
                "CARD.GRAND_FINALE".
        Output: the matching nocab_uuid, or None if no card is
            registered under spire_codex's alias for the stripped id.
        Side effects: none.
        Exceptions: none.
        """
        stripped_id = raw_card_id.removeprefix(_STS_GG_CARD_ID_PREFIX)
        card = self._card_binder.get_by_alias(
            GameId.SLAY_THE_SPIRE_2, DataSource.SPIRE_CODEX, stripped_id
        )
        return card.nocab_uuid if card is not None else None


def scan_runs_jsonl(raw_path: Path, metric: DeckOutcomeMetric) -> Path:
    """Drive metric.accumulate() over every run in a runs.jsonl file,
    then finalize it.

    Kept separate from DeckOutcomeMetric itself: "how to read sts_gg's
    JSONL" is a different concern from "what this metric computes" (see
    module docstring's STREAMING, NOT ACCUMULATION section). Not yet
    promoted to a shared sts_gg-level scanner module — single consumer
    today, same rule-of-three convention this project already applies
    elsewhere (see src/data_retrieval/README.md's download_to_file.py/
    rate_limiter.py extraction rationale) — worth revisiting only once
    a second sts_gg streaming metric needs the same driving loop.

    Inputs:
        raw_path: path to a runs.jsonl file (one JSON object per line —
            e.g. data/raw/sts_gg/runs.jsonl).
        metric: the DeckOutcomeMetric instance to drive. Its finalize()
            is called by this function (in a finally block — see
            below) — callers should not call it again themselves
            (finalize() is idempotent, so a redundant call is harmless,
            just unnecessary).
    Output: metric.finalize()'s return value (the output path).
    Side effects: reads raw_path; whatever metric.accumulate()/
        finalize() do (see those methods).
    Exceptions: raises if raw_path doesn't exist, or if any line isn't
        valid JSON. Whatever metric.accumulate() raises for a
        malformed run (see its own Exceptions) propagates — this
        function has no per-line best-effort skip of its own — but
        metric.finalize() still runs first (via try/finally), so the
        underlying ParquetWriter is always closed rather than left open
        on a mid-scan failure, even though the resulting file is then
        missing every run from the failure point onward.

    Example:
        >>> card_binder = CardBinder.load(
        ...     [Path("data/final/cards/slay_the_spire_2.jsonl")]
        ... )
        >>> metric = DeckOutcomeMetric(card_binder)
        >>> scan_runs_jsonl(Path("data/raw/sts_gg/runs.jsonl"), metric)
        PosixPath('data/final/metrics/sts_gg/deck_outcome.parquet')
    """
    try:
        with open(raw_path, "r", encoding="utf-8") as raw_file:
            for line in raw_file:
                if not line.strip():
                    continue
                row = json.loads(line)
                metric.accumulate(row)
    finally:
        output_path = metric.finalize()

    return output_path
