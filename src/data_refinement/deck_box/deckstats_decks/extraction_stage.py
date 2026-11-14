"""Deckstats decks -> the MTG DeckBox: the shared loop in
mtg_crawls/extraction_stage.py with this site's deck_parser.py."""

from pathlib import Path
from typing import ClassVar

from src.data_refinement.deck_box.deckstats_decks.deck_parser import DeckstatsDeckParser
from src.data_refinement.deck_box.mtg_crawls.extraction_stage import (
    CrawledMtgDeckExtractionStage,
)
from src.schema.data_source import DataSource


class DeckstatsDeckExtractionStage(CrawledMtgDeckExtractionStage):
    """Deckstats crawl shards -> GenericDecks (MTG)."""

    DATA_SOURCE: ClassVar[DataSource] = DataSource.DECKSTATS
    DEFAULT_RAW_PATH: ClassVar[Path] = Path("data/raw/deckstats")

    def __init__(self) -> None:
        """
        Inputs: none. Output: none (constructor).
        Side effects: none. Exceptions: none.
        """
        super().__init__(DeckstatsDeckParser())
