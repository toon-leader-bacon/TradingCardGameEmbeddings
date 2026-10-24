"""RunSampler: a background thread writing what a run holds every few
seconds, to a samples.csv (beside the run profile's CSVs).

Columns, fixed when start() creates the file:

    t                     seconds on the run profile's clock
                          (SampledSources.seconds_since_open)
    steps_attempted       the trainer's running count of step attempts
    vram_allocated_gib    torch's allocated, reserved and peak-allocated
    vram_reserved_gib     CUDA memory (0 without a GPU)
    vram_peak_gib
    cache_texts           the feature cache's running totals
    cache_encoded         (FeatureCacheStats); empty without a cache
    cache_admitted
    cache_uncacheable
    <tier>_hits           per tier of the cache at start(), named by
    <tier>_rows           TierStats.name (e.g. vram_hits, ram_rows,
    <tier>_pinned         disk_evicted); empty without a cache
    <tier>_evicted

System RAM, GPU engine utilization and CPU use are not here: the venv has
no psutil, so an experiment's own Windows counter samplers record them.

Reading the cache's totals between rows gives a hit rate over time:
the share of texts asked in an interval that were not encoded,
1 - d(cache_encoded) / d(cache_texts).
"""

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

from src.encoder_model.feature_cache.pooled_feature_cache import FeatureCacheStats
from src.training.recording.csv_table import CsvTable

_logger = logging.getLogger(__name__)

# How long the sampler waits between rows by default
DEFAULT_SAMPLE_SECONDS = 10.0

_GIB = 1024**3
_BASE_FIELDS = [
    "t",
    "steps_attempted",
    "vram_allocated_gib",
    "vram_reserved_gib",
    "vram_peak_gib",
    "cache_texts",
    "cache_encoded",
    "cache_admitted",
    "cache_uncacheable",
]
_TIER_SUFFIXES = ("hits", "rows", "pinned", "evicted")


@dataclass(frozen=True)
class SampledSources:
    """What RunSampler reads each interval.

    seconds_since_open: the run profile's clock (CsvRunProfile), so rows
        line up with steps.csv. cache_stats: the model's feature cache
        totals, None without a cache (CardEncoderModel.feature_cache_stats).
    steps_attempted: the trainer's running count of step attempts.
    """

    seconds_since_open: Callable[[], float]
    cache_stats: Callable[[], FeatureCacheStats | None]
    steps_attempted: Callable[[], int]


class RunSampler:
    """Writes samples.csv on a daemon thread until stopped."""

    def __init__(
        self,
        path: Path,
        sources: SampledSources,
        interval_seconds: float = DEFAULT_SAMPLE_SECONDS,
    ) -> None:
        """Inputs: path (the CSV; must not exist), sources,
            interval_seconds (> 0).
        Output: none. Side effects: none until start.
        Exceptions: ValueError if interval_seconds <= 0.

        Example:
            >>> sampler = RunSampler(profile_dir / "samples.csv", sources)
        """
        if interval_seconds <= 0:
            raise ValueError(f"interval_seconds must be > 0, got {interval_seconds}")
        self._path = path
        self._sources = sources
        self._interval = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._table: CsvTable | None = None
        self._tier_names: list[str] = []
        self._failed_once = False

    def start(self) -> None:
        """Fix the columns from the cache's tiers, write a first row now,
        then one per interval on a daemon thread.

        Inputs: none. Output: none.
        Side effects: creates the CSV; starts the thread.
        Exceptions: RuntimeError if already started; FileExistsError,
            OSError from creating the file.

        Example:
            >>> sampler.start()
        """
        if self._thread is not None:
            raise RuntimeError("RunSampler already started")
        # The tier columns are the cache's tiers now; a cacheless run has none
        stats = self._sources.cache_stats()
        self._tier_names = [] if stats is None else [tier.name for tier in stats.tiers]
        tier_fields = [
            f"{name}_{suffix}" for name in self._tier_names for suffix in _TIER_SUFFIXES
        ]
        self._table = CsvTable(self._path, _BASE_FIELDS + tier_fields)
        self._write_row()
        self._thread = threading.Thread(
            target=self._run, name="run-sampler", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Write a last row, stop the thread and close the file.
        Idempotent; nothing before start.

        Inputs: none. Output: none. Side effects: joins the thread.
        Exceptions: none (a failing last row is logged).

        Example:
            >>> sampler.stop()
        """
        if self._thread is None or self._table is None or self._stop.is_set():
            return
        self._stop.set()
        self._thread.join()
        self._write_row()
        self._table.close()

    def _run(self) -> None:
        """The thread's loop: a row per interval until stopped.

        Private helper - single consumer is start. Inputs: none.
        Output: none. Side effects: appends rows. Exceptions: none.
        """
        while not self._stop.wait(self._interval):
            self._write_row()

    def _write_row(self) -> None:
        """Write one sample; a failure is logged (the first at warning,
        later ones at debug) and the row skipped.

        Private helper - consumers are start, _run and stop.
        Inputs: none. Output: none. Side effects: appends a row; may log.
        Exceptions: none.
        """
        assert self._table is not None  # start created it before any row
        try:
            self._table.write(self._row())
        except Exception:
            log = _logger.debug if self._failed_once else _logger.warning
            self._failed_once = True
            log("run sampler skipped a row", exc_info=True)

    def _row(self) -> dict[str, object]:
        """One sample keyed by the CSV's columns; tiers the cache no longer
        reports, or a cacheless run, leave their columns empty.

        Private helper - single consumer is _write_row.
        Inputs: none. Output: dict. Side effects: reads the sources.
        Exceptions: whatever a source raises.
        """
        result: dict[str, object] = {
            "t": self._sources.seconds_since_open(),
            "steps_attempted": self._sources.steps_attempted(),
            **_vram_fields(),
        }
        stats = self._sources.cache_stats()
        if stats is None:
            return result

        # The cache's running totals, then each known tier's
        result.update(
            cache_texts=stats.texts,
            cache_encoded=stats.encoded,
            cache_admitted=stats.admitted,
            cache_uncacheable=stats.uncacheable,
        )
        for tier in stats.tiers:
            if tier.name not in self._tier_names:
                continue
            result[f"{tier.name}_hits"] = tier.hits
            result[f"{tier.name}_rows"] = tier.rows_used
            result[f"{tier.name}_pinned"] = tier.pinned_rows
            result[f"{tier.name}_evicted"] = tier.evicted
        return result


def _vram_fields() -> dict[str, float]:
    """torch's allocated, reserved and peak-allocated CUDA memory in GiB
    (zeros without a GPU).

    Private helper - single consumer is RunSampler._row.
    Inputs: none. Output: dict. Side effects: none. Exceptions: none.
    """
    if not torch.cuda.is_available():
        return {
            "vram_allocated_gib": 0.0,
            "vram_reserved_gib": 0.0,
            "vram_peak_gib": 0.0,
        }
    return {
        "vram_allocated_gib": torch.cuda.memory_allocated() / _GIB,
        "vram_reserved_gib": torch.cuda.memory_reserved() / _GIB,
        "vram_peak_gib": torch.cuda.max_memory_allocated() / _GIB,
    }
