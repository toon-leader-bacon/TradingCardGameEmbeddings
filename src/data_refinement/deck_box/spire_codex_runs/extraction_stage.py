"""Translates spire-codex's run export into Slay the Spire 2 decks.

RAW SHAPE: data/raw/spire_codex/runs/page_NNNNN.jsonl.gz, written by
src/data_retrieval/spire_codex/run_downloader.py - one run per line, the
game's own run record. Same schema as sts2runs' dump (players[].deck of
{"id": "CARD.<NAME>", ...}), except the run's id is "run_hash" (a hex
string) rather than "_serverId". So this stage is
Sts2RunsDeckExtractionStage with that key, its own deck namespace and
data source, over every page file (raw_files()).

WHAT IS KEPT: one final deck per player of every run, by default. The
project favors data quantity over quality, so abandoned runs
(was_abandoned, ~16% of the 2026-09 export) are kept too, though a quit
run's deck is not a finished one and early quits are mostly starter
decks. keep_abandoned=False drops them; a finer filter (drop only runs
abandoned before the first boss) is noted in deck_box/TODO.md. About 870
of the 1.67M runs also appear in sts2runs' dump; they are stored twice
(under different deck uuids).
"""

from pathlib import Path
from typing import ClassVar
from uuid import UUID

from src.data_refinement.deck_box.sts2runs.extraction_stage import (
    Sts2RunsDeckExtractionStage,
)
from src.data_retrieval.spire_codex.run_downloader import SpireCodexRunDownloader
from src.schema.data_source import DataSource

_PAGE_GLOB = "page_*.jsonl.gz"


class SpireCodexRunsDeckExtractionStage(Sts2RunsDeckExtractionStage):
    """spire-codex run export pages -> final GenericDecks (StS2)."""

    DEFAULT_RAW_PATH: ClassVar[Path] = SpireCodexRunDownloader.DEFAULT_RAW_DATA_DIR
    RUN_ID_KEY: ClassVar[str] = "run_hash"
    DECK_DATA_SOURCE: ClassVar[DataSource] = DataSource.SPIRE_CODEX
    # Fixed, arbitrary - never regenerate (deck identity depends on it)
    DECK_NAMESPACE: ClassVar[UUID] = UUID("3b7e1d52-9c4a-4f06-b8d1-6e2a0c9f5d17")
    DECK_NAME_PREFIX: ClassVar[str] = "spire_codex run"

    def __init__(self, keep_abandoned: bool = True) -> None:
        """Inputs: keep_abandoned (store the decks of abandoned runs).
        Side effects: none. Exceptions: none."""
        self._keep_abandoned = keep_abandoned

    def raw_files(self, raw_path: Path | None) -> list[Path]:
        """Every page file under raw_path, in name order (or raw_path
        itself when it is one page file).

        Inputs: raw_path (the pages directory, or one page file;
            DEFAULT_RAW_PATH when None).
        Output: list[Path] of page files.
        Side effects: lists the directory.
        Exceptions: FileNotFoundError if raw_path is missing or holds no
            page file.

        Example:
            >>> SpireCodexRunsDeckExtractionStage().raw_files(None)[0].name
            'page_00000.jsonl.gz'
        """
        path = raw_path or self.DEFAULT_RAW_PATH
        pages = [path] if path.is_file() else sorted(path.glob(_PAGE_GLOB))
        if not pages:
            raise FileNotFoundError(f"no {_PAGE_GLOB} run pages at {path}")
        return pages

    def _keeps_run(self, row: dict) -> bool:
        """False for an abandoned run when keep_abandoned is off (see the
        module docstring).
        Inputs: row. Output: bool. Side effects: none. Exceptions: none."""
        return self._keep_abandoned or not row.get("was_abandoned", False)
