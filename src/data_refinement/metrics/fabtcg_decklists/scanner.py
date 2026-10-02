"""Drives every fabtcg_decklists metric over one pass of the raw decklist
directory (data/raw/fabtcg_decklists/decklists/<slug>.html).

A row is {"slug": <file stem>}: the slug names the decklist, and every
metric here reads that deck's cards from the published deck box by slug
(published_decklists.py), so the fragment HTML itself is not parsed
again. Same shape as ../play_gwent/scanner.py (a per-container copy, per
this project's convention of deferring cross-container dedup): one
metric's accumulate()/finalize() failure is logged, never raised, so the
others still run. Never touches a deck box.
"""

import logging
from pathlib import Path

from src.data_refinement.metrics.metric import Metric
from src.data_retrieval.fabtcg_decklists.downloader import FabtcgDecklistDownloader

_logger = logging.getLogger(__name__)

DEFAULT_RAW_PATH: Path = FabtcgDecklistDownloader.DEFAULT_RAW_DATA_DIR / "decklists"


def scan_decklist_files(raw_dir: Path, metrics: list[Metric[dict]]) -> None:
    """Feed one {"slug": stem} row per raw decklist file to every metric
    (sorted by file name), then finalize all of them.

    Inputs:
        raw_dir: a directory of <slug>.html decklist files.
        metrics: every Metric[dict] to drive.
    Output: none.
    Side effects: lists raw_dir; calls accumulate() per row and
        finalize() once on every metric; logs (logging.exception) a
        metric's accumulate()/finalize() failure instead of raising.
    Exceptions: FileNotFoundError if raw_dir does not exist.

    Example:
        >>> metrics = [HeroMaskedFromDeckMetric(binder, published_box)]
        >>> scan_decklist_files(DEFAULT_RAW_PATH, metrics)
    """
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"no decklist directory at {raw_dir}")
    for fragment_path in sorted(raw_dir.glob("*.html")):
        row = {"slug": fragment_path.stem}
        for metric in metrics:
            try:
                metric.accumulate(row)
            except Exception:
                _logger.exception(
                    "scan_decklist_files: %r raised on %s - skipping that row "
                    "for that metric",
                    metric,
                    fragment_path.name,
                )
    for metric in metrics:
        try:
            metric.finalize()
        except Exception:
            _logger.exception(
                "scan_decklist_files: %r raised from finalize() - its output may "
                "be missing or incomplete",
                metric,
            )
