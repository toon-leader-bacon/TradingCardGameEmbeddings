"""Drives a shared read pass over one or more isotropic Flavor B
("full game log") tarballs across multiple Metric instances at once -
the games/ analogue of ../summary/scanner.py, adapted for THIS
source's on-disk shape: each raw archive is a bare `YYYYMMDD.tar.bz2`
(see ../BRAINSTORM.md - not `*-summary.tar.bz2`) containing one HTML
file per individual game, not one JSONL member per day.

Same failure-isolation contract as ../summary/scanner.py's
scan_isotropic_summary_archives(): one metric's bug, or one
unparseable HTML member, must not silently stop a different,
still-healthy metric (or the rest of the archive) from being seen.
DECK BOX IS NOT THIS FILE'S CONCERN, for the same reason as every
other scanner in this project - a metric that takes a DeckBox writes
into it during accumulate() itself.

TWO ENTRY POINTS, ONE SHARED MEMBER-ITERATION HELPER:
scan_isotropic_game_log_archives() (header-only, Metric[GameHeader] -
this container's original six metrics) and
scan_isotropic_game_logs_archives() (full-body, Metric[GameLog] - the
mid-game-state metric family) both read the exact same archives member
by member; only the parse step at the end differs (parse_game_header()
vs. game_log_parser.parse_game_log()). _iter_archive_members() below is
the one place that tarfile-iteration logic lives now, consumed by both
entry points' own per-row iterator.
"""

import logging
import tarfile
from pathlib import Path
from typing import Iterator, TypeVar

from tqdm import tqdm

from src.data_refinement.metrics.isotropic.games.game_log_parser import (
    GameLog,
    parse_game_log,
)
from src.data_refinement.metrics.isotropic.games.header_parser import (
    GameHeader,
    parse_game_header,
)
from src.data_refinement.metrics.metric import Metric

_logger = logging.getLogger(__name__)

_RawRowT = TypeVar("_RawRowT")


def scan_isotropic_game_log_archives(
    archive_paths: list[Path], metrics: list[Metric[GameHeader]]
) -> None:
    """Drive every metric in `metrics` over every parseable GameHeader
    across every HTML member of every archive in archive_paths, then
    finalize all of them.

    Inputs:
        archive_paths: every bare `YYYYMMDD.tar.bz2` full-log archive
            to scan (e.g. both
            data/raw/isotropic/2010_20101011.tar.bz2 and
            data/raw/isotropic/2013_20130315.tar.bz2, to pool both
            surviving days into one training corpus) -
            `201010_11_all.tar.bz2` is a byte-identical duplicate of
            the first (../BRAINSTORM.md) and should never be passed
            alongside it.
        metrics: every Metric[GameHeader] to drive over this combined
            read pass. Each metric's own accumulate()/finalize()
            failures are isolated from the others.
    Output: none.
    Side effects: reads every archive in archive_paths once, extracting
        each HTML member into memory one at a time; calls
        header_parser.parse_game_header() on each member's content,
        skipping (not feeding to any metric) a member that fails to
        parse - logged once, not per metric, since an unparseable file
        is this scanner's own concern, not a per-metric failure; calls
        accumulate() on every metric for every successfully parsed
        GameHeader, then finalize() on every metric. Prints one tqdm
        progress bar per archive to stderr, sized against that
        archive's member count.
    Exceptions: raises if any archive_paths entry doesn't exist or
        isn't a valid tar.bz2 - only a per-metric accumulate()/
        finalize() failure, or a single member's parse failure, is
        caught and isolated, not an archive-level failure.

    Example:
        >>> deck_box = DeckBox()  # metrics-private
        >>> metrics = [
        ...     OpeningBuyRateMetric(card_binder),
        ...     PileExhaustionRateMetric(card_binder),
        ... ]
        >>> scan_isotropic_game_log_archives(
        ...     [
        ...         Path("data/raw/isotropic/2010_20101011.tar.bz2"),
        ...         Path("data/raw/isotropic/2013_20130315.tar.bz2"),
        ...     ],
        ...     metrics,
        ... )
    """
    for archive_path in archive_paths:
        for header in _iter_game_headers(archive_path):
            for metric in metrics:
                _accumulate_isolated(metric, header)

    for metric in metrics:
        _finalize_isolated(metric)


def scan_isotropic_game_logs_archives(
    archive_paths: list[Path], metrics: list[Metric[GameLog]]
) -> None:
    """Drive every metric in `metrics` over every parseable GameLog
    (full turn-by-turn body, not just the header) across every HTML
    member of every archive in archive_paths, then finalize all of
    them.

    Inputs:
        archive_paths: every bare `YYYYMMDD.tar.bz2` full-log archive
            to scan - identical set of valid inputs as
            scan_isotropic_game_log_archives().
        metrics: every Metric[GameLog] to drive over this combined read
            pass (the mid-game-state metric family - a different row
            type than scan_isotropic_game_log_archives()'s
            Metric[GameHeader] metrics, so the two entry points are
            never called with the same metrics list).
    Output: none.
    Side effects: reads every archive in archive_paths once; calls
        game_log_parser.parse_game_log() on each member, skipping a
        member that fails to parse (logged once, not per metric); calls
        accumulate() on every metric for every successfully parsed
        GameLog, then finalize() on every metric. Prints one tqdm
        progress bar per archive to stderr.
    Exceptions: raises if any archive_paths entry doesn't exist or
        isn't a valid tar.bz2 - only a per-metric accumulate()/
        finalize() failure, or a single member's parse failure, is
        caught and isolated, not an archive-level failure.

    Example:
        >>> deck_box = DeckBox()  # metrics-private
        >>> metrics = [NextBuyPredictionMetric(card_binder, deck_box)]
        >>> scan_isotropic_game_logs_archives(
        ...     [Path("data/raw/isotropic/2013_20130315.tar.bz2")],
        ...     metrics,
        ... )
    """
    for archive_path in archive_paths:
        for game_log in _iter_game_logs(archive_path):
            for metric in metrics:
                _accumulate_isolated(metric, game_log)

    for metric in metrics:
        _finalize_isolated(metric)


def _iter_archive_members(archive_path: Path) -> Iterator[tuple[str, str]]:
    """Yield every regular-file member's (name, decoded content) across
    one full-log tarball.

    Private helper - shared by _iter_game_headers() and
    _iter_game_logs() (module docstring's TWO ENTRY POINTS note).
    Deliberately extracts one member into memory at a time (never the
    whole archive) - the larger of the two surviving days has 10,815
    members.

    Inputs:
        archive_path: one bare `YYYYMMDD.tar.bz2` file.
    Output: a generator of (member_name, html_text) pairs, one per
        regular-file member - a non-regular-file tarinfo entry (e.g. a
        directory) is skipped, not yielded.
    Side effects: opens and reads archive_path; prints one tqdm
        progress bar to stderr, sized against this archive's member
        count.
    Exceptions: whatever tarfile.open() raises on a malformed archive.
    """
    with tarfile.open(archive_path, "r:bz2") as archive:
        members = archive.getmembers()
        for member in tqdm(
            members, desc=f"isotropic scan: {archive_path.name}", unit="game"
        ):
            member_file = archive.extractfile(member)
            if member_file is None:
                # A non-regular-file tarinfo entry (e.g. a directory) -
                # nothing to parse.
                continue
            yield member.name, member_file.read().decode("utf-8", errors="replace")


def _iter_game_headers(archive_path: Path) -> Iterator[GameHeader]:
    """Yield every successfully-parsed GameHeader across every HTML
    member of one full-log tarball.

    Private helper - single consumer is scan_isotropic_game_log_archives().

    Inputs:
        archive_path: one bare `YYYYMMDD.tar.bz2` file.
    Output: a generator of GameHeader, one per member that
        _parse_member_isolated() returns a real GameHeader for. A
        member is skipped, not yielded, for two DIFFERENT reasons this
        function does not distinguish between: parse_game_header()
        itself returning None (header_parser.py's own RESIGNED/NO-
        WINNER GAMES note - an expected, common, unlogged outcome), or
        parse_game_header() raising on a genuinely malformed member
        (see _parse_member_isolated() - logged, not silent, and
        distinct from the first case).
    Side effects: whatever _iter_archive_members() does, plus whatever
        _parse_member_isolated() does per member (at most one
        logging.exception() call per malformed member).
    Exceptions: whatever _iter_archive_members() raises - a single
        member's parse EXCEPTION is caught and isolated by
        _parse_member_isolated(), never propagated from here.
    """
    for member_name, html_text in _iter_archive_members(archive_path):
        header = _parse_header_member_isolated(member_name, html_text)
        if header is not None:
            yield header


def _iter_game_logs(archive_path: Path) -> Iterator[GameLog]:
    """Yield every successfully-parsed GameLog across every HTML member
    of one full-log tarball.

    Private helper - single consumer is scan_isotropic_game_logs_archives().
    Structurally identical to _iter_game_headers() but for
    game_log_parser.parse_game_log() instead of
    header_parser.parse_game_header() - kept as its own function, not a
    shared generic over "which parse function" (that would need a
    function-object parameter typed loosely enough to accept either
    parser, trading real type safety for saving two thin function
    bodies - not a good trade per PRINCIPLES.md's correctness-first
    priority).

    Inputs:
        archive_path: one bare `YYYYMMDD.tar.bz2` file.
    Output: a generator of GameLog, one per member that
        _parse_log_member_isolated() returns a real GameLog for -
        skipped for the same two reasons _iter_game_headers() documents
        (an ordinary None return vs. a logged exception).
    Side effects: whatever _iter_archive_members() does, plus whatever
        _parse_log_member_isolated() does per member.
    Exceptions: whatever _iter_archive_members() raises.
    """
    for member_name, html_text in _iter_archive_members(archive_path):
        game_log = _parse_log_member_isolated(member_name, html_text)
        if game_log is not None:
            yield game_log


def _parse_header_member_isolated(
    member_name: str, html_text: str
) -> GameHeader | None:
    """Call header_parser.parse_game_header(html_text), logging (not
    raising) if it raises.

    Private helper - single consumer is _iter_game_headers(). This is
    what makes good on parse_game_header()'s own docstring promise that
    "a caller driving many files (../scanner.py) is expected to
    isolate" a per-file parse exception - the _accumulate_isolated()/
    _finalize_isolated() shape, one level earlier in the pipeline
    (structural parse failures happen before any metric ever sees a
    row, unlike an accumulate()/finalize() failure).

    Inputs:
        member_name: the archive member's own name, for the log
            message only (e.g. "game-20130315-000003-b7c5dc64.html").
        html_text: that member's raw content.
    Output: parse_game_header(html_text)'s own return value on success
        (a GameHeader, or None for the module docstring's ordinary
        RESIGNED/NO-WINNER GAMES case) - or None if parse_game_header()
        raised, a DIFFERENT "skip this member" outcome that is logged,
        unlike the ordinary None case.
    Side effects: none on success; on a raised exception, emits one
        logging.exception() call naming member_name, instead of
        propagating.
    Exceptions: none - every exception from parse_game_header() is
        caught and logged here.
    """
    try:
        return parse_game_header(html_text)
    except Exception:
        _logger.exception(
            "scan_isotropic_game_log_archives: parse_game_header() raised "
            "on member %r - skipping just that member",
            member_name,
        )
        return None


def _parse_log_member_isolated(member_name: str, html_text: str) -> GameLog | None:
    """Call game_log_parser.parse_game_log(html_text), logging (not
    raising) if it raises.

    Private helper - single consumer is _iter_game_logs(). Mirrors
    _parse_header_member_isolated()'s own isolation contract exactly,
    for parse_game_log() instead of parse_game_header().

    Inputs:
        member_name: the archive member's own name, for the log
            message only.
        html_text: that member's raw content.
    Output: parse_game_log(html_text)'s own return value on success, or
        None if parse_game_log() raised, a DIFFERENT "skip this member"
        outcome that is logged, unlike an ordinary None return.
    Side effects: none on success; on a raised exception, emits one
        logging.exception() call naming member_name, instead of
        propagating.
    Exceptions: none - every exception from parse_game_log() is caught
        and logged here.
    """
    try:
        return parse_game_log(html_text)
    except Exception:
        _logger.exception(
            "scan_isotropic_game_logs_archives: parse_game_log() raised "
            "on member %r - skipping just that member",
            member_name,
        )
        return None


def _accumulate_isolated(metric: Metric[_RawRowT], row: _RawRowT) -> None:
    """Call metric.accumulate(row), logging (not raising) on failure.

    Private helper - shared by both scan_isotropic_game_log_archives()
    (row: GameHeader) and scan_isotropic_game_logs_archives() (row:
    GameLog) - generic over the raw row type since this isolation
    contract has nothing type-specific in it, unlike
    _parse_header_member_isolated()/_parse_log_member_isolated() above
    (which genuinely differ by which parse function they call). Mirrors
    ../summary/scanner.py's own _accumulate_isolated() - kept as this
    scanner's own copy rather than a shared cross-scanner helper,
    matching this project's existing convention of one scanner per
    raw-source shape.

    Inputs:
        metric: the Metric[_RawRowT] to drive.
        row: one parsed row (GameHeader or GameLog) to feed it.
    Output: none.
    Side effects: whatever metric.accumulate() does on success; on
        failure, emits one logging.exception() call instead of
        propagating.
    Exceptions: none - every exception from metric.accumulate() is
        caught and logged here.
    """
    try:
        metric.accumulate(row)
    except Exception:
        _logger.exception(
            "isotropic games/ scanner: %r raised from accumulate() on one "
            "row - skipping just that row for this metric, continuing the "
            "scan for every other metric",
            metric,
        )


def _finalize_isolated(metric: Metric[_RawRowT]) -> None:
    """Call metric.finalize(), logging (not raising) on failure.

    Private helper - shared by both entry points, same reasoning as
    _accumulate_isolated() above. Mirrors _accumulate_isolated()'s
    isolation for the finalize phase.

    Inputs:
        metric: the Metric[_RawRowT] to finalize.
    Output: none.
    Side effects: whatever metric.finalize() does on success; on
        failure, emits one logging.exception() call instead of
        propagating.
    Exceptions: none - every exception from metric.finalize() is
        caught and logged here.
    """
    try:
        metric.finalize()
    except Exception:
        _logger.exception(
            "isotropic games/ scanner: %r raised from finalize() - its "
            "output may be missing or incomplete, but every other metric "
            "still finalizes normally",
            metric,
        )
