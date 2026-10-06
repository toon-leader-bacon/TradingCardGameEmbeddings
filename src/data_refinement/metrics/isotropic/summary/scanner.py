"""Drives a shared read pass over one or more isotropic Flavor A
("summary") tarballs across multiple Metric instances at once - the
isotropic analogue of ../../sts_gg/scanner.py's scan_runs_jsonl(), adapted
for this source's actual on-disk shape: each raw file is a
`*-summary.tar.bz2` archive (see BRAINSTORM.md) containing one
`games-YYYYMMDD.json` JSONL member per day, not one flat JSONL file.

Same failure-isolation contract as scan_runs_jsonl(): one metric's bug
must not silently stop a different, still-healthy metric in the same
list from seeing the rest of the archives. DECK BOX IS NOT THIS FILE'S
CONCERN, for the same reason scan_runs_jsonl() isn't either - a metric
that takes a DeckBox writes into it during accumulate() itself; this
function only calls accumulate()/finalize() on already-constructed
Metric[dict] instances.
"""

import json
import logging
import tarfile
from functools import partial
from pathlib import Path
from typing import Iterator

from tqdm import tqdm

from src.data_refinement.metrics.isolated_call import call_isolated
from src.data_refinement.metrics.metric import Metric

_logger = logging.getLogger(__name__)


def scan_isotropic_summary_archives(
    archive_paths: list[Path], metrics: list[Metric[dict]]
) -> None:
    """Drive every metric in `metrics` over every game row in every
    `games-YYYYMMDD.json` member of every archive in archive_paths,
    then finalize all of them.

    Inputs:
        archive_paths: every `*-summary.tar.bz2` file to scan (e.g.
            both data/raw/isotropic/2010_201012-summary.tar.bz2 and
            data/raw/isotropic/2013_201303-summary.tar.bz2, to pool
            both months into one training corpus) - order matters only
            for progress-bar sequencing, not for correctness.
        metrics: every Metric[dict] to drive over this combined read
            pass. Each metric's own accumulate()/finalize() failures
            are isolated from the others (mirrors scan_runs_jsonl()).
    Output: none.
    Side effects: reads every archive in archive_paths once, extracting
        each JSONL member into memory one at a time (never all members
        at once - see iter_summary_rows()); calls accumulate() on
        every metric for every row, then finalize() on every metric.
        Logs loudly on any per-metric accumulate()/finalize() failure,
        rather than raising. Prints one tqdm progress bar per archive
        to stderr, sized against that archive's member count.
    Exceptions: raises if any archive_paths entry doesn't exist or
        isn't a valid tar.bz2, or if any JSONL line inside a member
        isn't valid JSON - only a per-metric accumulate()/finalize()
        failure is caught and isolated, not an archive-level failure.

    Example:
        >>> deck_box = DeckBox()  # metrics-private
        >>> metrics = [
        ...     VetoRateMetric(card_binder),
        ...     FullDeckWinPredictionMetric(card_binder, deck_box),
        ... ]
        >>> scan_isotropic_summary_archives(
        ...     [
        ...         Path("data/raw/isotropic/2010_201012-summary.tar.bz2"),
        ...         Path("data/raw/isotropic/2013_201303-summary.tar.bz2"),
        ...     ],
        ...     metrics,
        ... )
        >>> deck_box.save(
        ...     Path("data/metrics/isotropic/deck_box.db"),
        ...     GameId.DOMINION,
        ...     card_binder.version_for(GameId.DOMINION),
        ... )
    """
    for archive_path in archive_paths:
        for row in iter_summary_rows(archive_path):
            for metric in metrics:
                call_isolated(
                    _logger,
                    metric,
                    "accumulate() on one row",
                    partial(metric.accumulate, row),
                )

    for metric in metrics:
        call_isolated(_logger, metric, "finalize()", metric.finalize)


def iter_summary_rows(archive_path: Path) -> Iterator[dict]:
    """Yield every parsed JSON game row across every `games-*.json`
    member of one summary tarball, one day-file at a time.

    Consumers: scan_isotropic_summary_archives() and
    deck_box/isotropic/extraction_stage.py (final decks).
    Deliberately extracts one member into memory at a time (never the
    whole archive) - the largest single member confirmed in
    BRAINSTORM.md is a day of ~18K rows, small enough to hold, but
    reading the whole tarball into memory upfront isn't.

    Inputs:
        archive_path: one `*-summary.tar.bz2` file.
    Output: a generator of dicts, one per non-blank JSONL line across
        every member, in tarfile.getmembers() order (day order, since
        BRAINSTORM.md confirms member names sort as games-YYYYMMDD.json).
    Side effects: opens and reads archive_path; prints one tqdm
        progress bar to stderr per archive (member-count-sized, per
        this module's own docstring).
    Exceptions: whatever tarfile.open()/json.loads() raise on a
        malformed archive or line.
    """
    with tarfile.open(archive_path, "r:bz2") as archive:
        members = archive.getmembers()
        for member in tqdm(
            members, desc=f"isotropic scan: {archive_path.name}", unit="day"
        ):
            member_file = archive.extractfile(member)
            if member_file is None:
                # A non-regular-file tarinfo entry (e.g. a directory) -
                # nothing to parse.
                continue
            for line in member_file.read().decode("utf-8").splitlines():
                if not line.strip():
                    continue
                yield json.loads(line)
