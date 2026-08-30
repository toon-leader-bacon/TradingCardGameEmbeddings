"""The v2 Metric Strategy interface — see plans/metrics_v2.md for the
full design this codifies.

Deliberately smaller than src/data_refinement/seventeenlands/
game_data_metrics/metric.py's Metric Protocol (the v1 interface, still
in active use, not modified by this file): no `resolved`/`card_columns`
parameter on accumulate() (every v2 metric resolves its own card
identifiers internally, via its own held CardBinder/source_game — see
each concrete metric's own docstring), no save_state()/load_state()/
checkpointing at all (a failed scan is simply rerun from row 0 — an
explicit, deliberate simplification, not an oversight), and finalize()
returns the Path it wrote its own output to rather than returning data
for a shared writer to merge (every v2 metric owns writing its own
output file, in whatever shape it wants).

Implemented as a typing.Protocol (structural typing), matching every
other Strategy interface in this project (CardIngestionStage,
SingleCardModel, MultiCardModel, Dojo, Loss) — not inheritance.
"""

from pathlib import Path
from typing import ClassVar, Protocol

import pandas as pd


class Metric(Protocol):
    """One pluggable metric-building Strategy, driven by CsvScanner
    (csv_scanner.py) or any future scanner over any future raw data
    shape.

    DEFAULT_OUTPUT_DIR/DEFAULT_OUTPUT_NAME name this metric's
    conventional output location, the same ClassVar-pair-plus-
    default_output_path()-staticmethod convention already used by
    CardBinder and every existing *Scanner in this project — NOT
    itself part of this Protocol (each concrete metric's own
    default_output_path() signature is parameterized however that
    metric needs, e.g. expansion/format_code, so it can't be pinned to
    one fixed shape here — same reason no *Scanner's own
    default_output_path() is protocol-constrained today either).
    """

    name: str
    DEFAULT_OUTPUT_DIR: ClassVar[Path]
    DEFAULT_OUTPUT_NAME: ClassVar[str]

    def accumulate(self, chunk: pd.DataFrame) -> None:
        """Process one chunk of raw data.

        Called once per chunk, in file order, by whichever scanner is
        driving this metric — possibly interleaved with other active
        metrics' own accumulate() calls over the same chunk, but never
        concurrently (one scanner, one pass, N metrics driven in
        sequence per chunk). Implementations decide for themselves
        whether this call updates in-memory running state (an
        aggregating metric) or performs this chunk's entire job
        immediately, including writing output (a streaming metric) —
        both are valid Metric implementations; nothing about this
        Protocol favors one shape over the other.

        Inputs:
            chunk: one chunk of a raw data file, shape/columns
                dependent on which raw source this metric reads —
                e.g. one chunk of a 17lands game_data CSV, read via
                pandas.read_csv(..., chunksize=...).
        Output: none.
        Side effects: implementation-defined — mutates internal state,
            writes to this metric's own output file, or both.
        Exceptions: implementation-defined.
        """
        ...

    def finalize(self) -> Path:
        """Finish this metric's work and report where its output landed.

        Called once, after every chunk this scan will ever see has
        already been passed to accumulate(). An aggregating metric
        turns its accumulated state into output and writes it here;
        a streaming metric (whose real work already happened per-chunk
        inside accumulate()) closes whatever it's been writing to and
        just reports the path.

        Inputs: none (uses this metric's own internal state).
        Output: the path this metric wrote its own output to — the
            caller (a scanner) never inspects what's inside, only that
            something landed at this path.
        Side effects: implementation-defined — typically writes to (or
            closes) this metric's own output file.
        Exceptions: implementation-defined.
        """
        ...
