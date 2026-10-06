"""Drives every sts2_runs metric over one read pass of each StS2 run
source (spire_codex's run pages, sts2runs' dump).

Reading reuses each source's deck box extraction stage (raw_files() and
runs()), so the raw files are found and decompressed the same way the
published deck box was built from them. Each raw run is parsed once
(run_parser.py), filtered once (scored_run(), the conditioning policy
below), and handed to every metric.

CONDITIONING (scored_run): a run is scored only if it is a standard-mode
game (daily and custom runs carry rule modifiers), not flagged
_isCheated, and not abandoned. Abandoned runs (~15% of spire_codex) are
neither a win nor a real loss, and an early quit leaves a near-starter
deck, so they would blur every outcome label. Within a scored run, a
player whose final deck has no aliased card (every slot is the Unknown
sentinel) is dropped; a run left with no player is not scored.

MALFORMED RUNS: a raw run the parser cannot read (e.g. a modded run
whose deck entries lack "floor_added_to_deck": 7 of 50k runs in page
00016) is logged, counted as MALFORMED and skipped; it never stops the
scan.

FAILURE ISOLATION: one metric's accumulate()/finalize() failure is
logged and never stops the others, via the shared call_isolated()
helper (../isolated_call.py).

The 868 runs present in both sources are scored twice (they are two
different decks in the deck box too); that is 0.05% of the runs.
"""

import dataclasses
import logging
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Sequence

from src.data_refinement.card_binder.card_lookup import CardLookup
from src.data_refinement.deck_box.spire_codex_runs.extraction_stage import (
    SpireCodexRunsDeckExtractionStage,
)
from src.data_refinement.deck_box.sts2runs.extraction_stage import (
    Sts2RunsDeckExtractionStage,
)
from src.data_refinement.metrics.isolated_call import call_isolated
from src.data_refinement.metrics.metric import Metric
from src.data_refinement.metrics.sts2_runs.run_parser import Sts2RunParser
from src.data_refinement.metrics.sts2_runs.run_record import RunOutcome, Sts2Run

_logger = logging.getLogger(__name__)

_SCORED_GAME_MODE = "standard"


@dataclass(frozen=True)
class RunSource:
    """One raw run source: its deck box extraction stage, and the raw
    path to read (None = the stage's DEFAULT_RAW_PATH)."""

    stage: Sts2RunsDeckExtractionStage
    raw_path: Path | None = None


class RunExclusion(Enum):
    """Why a run read was not scored (see CONDITIONING, MALFORMED RUNS)."""

    NOT_STANDARD = "not_standard"
    CHEATED = "cheated"
    ABANDONED = "abandoned"
    NO_KNOWN_CARDS = "no_known_cards"
    MALFORMED = "malformed"


@dataclass(frozen=True)
class ScanTally:
    """Runs read: how many were scored, and how many were excluded for
    each RunExclusion."""

    scored: int
    excluded: Counter[RunExclusion]

    def as_dict(self) -> dict[str, int]:
        """{"scored": n, <exclusion value>: n, ...}, for logs.
        Inputs: none. Output: dict[str, int]. Side effects: none.
        Exceptions: none."""
        return {
            "scored": self.scored,
            **{reason.value: count for reason, count in self.excluded.items()},
        }


def scan_sts2_runs(
    sources: Sequence[RunSource],
    card_lookup: CardLookup,
    metrics: Sequence[Metric[Sts2Run]],
) -> ScanTally:
    """Feed every scored run of every source to every metric, then
    finalize them all.

    Inputs:
        sources: the run sources to read, in order.
        card_lookup: the Slay the Spire 2 binder (read only).
        metrics: Metric[Sts2Run]s (deck_label_metrics.py,
            card_average_metrics.py).
    Output: ScanTally over every run read.
    Side effects: reads every source's raw files (one progress bar per
        file); calls accumulate() per scored run and finalize() once on
        every metric (they write their outputs); logs malformed runs,
        per-metric failures and one INFO line with the tally.
    Exceptions: whatever reading a raw file raises (a raw-file failure
        is not isolated; a malformed run and a metric failure are).

    Example:
        >>> binder = CardBinder.load([CardBinder.default_output_path(GameId.SLAY_THE_SPIRE_2)])
        >>> metrics = [WinMetric(binder), CardWinRateMetric(binder)]
        >>> scan_sts2_runs(default_run_sources(), binder, metrics).scored
    """
    scored = 0
    excluded: Counter[RunExclusion] = Counter()
    for source in sources:
        parser = Sts2RunParser(card_lookup, source.stage)
        for path in source.stage.raw_files(source.raw_path):
            for row in source.stage.runs(path):
                verdict = _scored_or_excluded(parser, source.stage, row)
                if isinstance(verdict, RunExclusion):
                    excluded[verdict] += 1
                    continue
                scored += 1
                for metric in metrics:
                    call_isolated(
                        _logger,
                        metric,
                        f"accumulate() on run {verdict.run_id}",
                        partial(metric.accumulate, verdict),
                    )

    for metric in metrics:
        call_isolated(_logger, metric, "finalize()", metric.finalize)
    tally = ScanTally(scored, excluded)
    _logger.info("scan_sts2_runs: %s", tally.as_dict())
    return tally


def scored_run(run: Sts2Run) -> Sts2Run | RunExclusion:
    """Apply the CONDITIONING policy (module docstring) to one run.

    Inputs: run (Sts2Run).
    Output: the run with only its players that have an aliased card, or
        the RunExclusion that keeps it out.
    Side effects: none (run is not modified; a new Sts2Run is built).
    Exceptions: none.

    Example:
        >>> scored_run(abandoned_run)
        <RunExclusion.ABANDONED: 'abandoned'>
    """
    if run.game_mode != _SCORED_GAME_MODE:
        return RunExclusion.NOT_STANDARD
    if run.is_cheated:
        return RunExclusion.CHEATED
    if run.outcome is RunOutcome.ABANDONED:
        return RunExclusion.ABANDONED
    players = tuple(player for player in run.players if player.known_card_uuids())
    if not players:
        return RunExclusion.NO_KNOWN_CARDS
    return dataclasses.replace(run, players=players)


def default_run_sources() -> tuple[RunSource, ...]:
    """Both StS2 run sources at their default raw paths: spire_codex's
    run pages, then sts2runs' dump.

    Inputs: none. Output: tuple[RunSource, ...]. Side effects: none.
    Exceptions: none.

    Example:
        >>> [type(s.stage).__name__ for s in default_run_sources()]
        ['SpireCodexRunsDeckExtractionStage', 'Sts2RunsDeckExtractionStage']
    """
    return (
        RunSource(SpireCodexRunsDeckExtractionStage()),
        RunSource(Sts2RunsDeckExtractionStage()),
    )


def _scored_or_excluded(
    parser: Sts2RunParser, stage: Sts2RunsDeckExtractionStage, row: dict
) -> Sts2Run | RunExclusion:
    """Parse one raw run and apply scored_run(); a run the parser cannot
    read is RunExclusion.MALFORMED (logged), never an error.

    Inputs: parser, stage (for the run id key, in the log line), row.
    Output: Sts2Run | RunExclusion.
    Side effects: logs a warning for a malformed run. Exceptions: none.
    """
    try:
        run = parser.parse(row)
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        _logger.warning(
            "scan_sts2_runs: skipping malformed run %r (%r)",
            row.get(stage.RUN_ID_KEY),
            error,
        )
        return RunExclusion.MALFORMED
    return scored_run(run)
