"""Moxfield decks -> the MTG DeckBox: the shared loop in
mtg_crawls/extraction_stage.py with this site's deck_parser.py."""

from pathlib import Path
from typing import ClassVar

from src.data_refinement.deck_box.moxfield_decks.deck_parser import MoxfieldDeckParser
from src.data_refinement.deck_box.mtg_crawls.extraction_stage import (
    CrawledMtgDeckExtractionStage,
)
from src.schema.data_source import DataSource


class MoxfieldDeckExtractionStage(CrawledMtgDeckExtractionStage):
    """Moxfield crawl shards -> GenericDecks (MTG)."""

    DATA_SOURCE: ClassVar[DataSource] = DataSource.MOXFIELD
    DEFAULT_RAW_PATH: ClassVar[Path] = Path("data/raw/moxfield")

    def __init__(self) -> None:
        """
        Inputs: none. Output: none (constructor).
        Side effects: none. Exceptions: none.
        """
        super().__init__(MoxfieldDeckParser())
