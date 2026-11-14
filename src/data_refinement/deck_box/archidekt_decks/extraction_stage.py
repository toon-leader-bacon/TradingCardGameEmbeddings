"""Archidekt decks -> the MTG DeckBox: the shared loop in
mtg_crawls/extraction_stage.py with this site's deck_parser.py."""

from pathlib import Path
from typing import ClassVar

from src.data_refinement.deck_box.archidekt_decks.deck_parser import ArchidektDeckParser
from src.data_refinement.deck_box.mtg_crawls.extraction_stage import (
    CrawledMtgDeckExtractionStage,
)
from src.schema.data_source import DataSource


class ArchidektDeckExtractionStage(CrawledMtgDeckExtractionStage):
    """Archidekt crawl shards -> GenericDecks (MTG)."""

    DATA_SOURCE: ClassVar[DataSource] = DataSource.ARCHIDEKT
    DEFAULT_RAW_PATH: ClassVar[Path] = Path("data/raw/archidekt")

    def __init__(self, include_unlisted: bool = False) -> None:
        """
        Inputs: include_unlisted (keep decks their authors left
            unlisted; default False, for published training data).
        Output: none (constructor). Side effects: none.
        Exceptions: none.
        """
        super().__init__(ArchidektDeckParser(include_unlisted))
