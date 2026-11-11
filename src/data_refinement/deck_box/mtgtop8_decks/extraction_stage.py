"""MTGTop8 decks -> the MTG DeckBox: the shared loop in
mtg_crawls/extraction_stage.py with this site's deck_parser.py.

The crawl keeps one directory per format (data/raw/mtgtop8/<code>/,
each a JsonlShards directory), so this stage reads every format
directory's shards, in format-code order, through the shared loop's
shard_rows hook.
"""

from pathlib import Path
from typing import ClassVar, Iterator

from src.data_refinement.deck_box.mtg_crawls.extraction_stage import (
    CrawledMtgDeckExtractionStage,
)
from src.data_refinement.deck_box.mtgtop8_decks.deck_parser import MtgTop8DeckParser
from src.data_retrieval.resumable_crawl.crawl_files import read_shard_rows
from src.schema.data_source import DataSource
from src.schema.json_types import JsonObject


class MtgTop8DeckExtractionStage(CrawledMtgDeckExtractionStage):
    """MTGTop8 crawl shards (every format) -> GenericDecks (MTG)."""

    DATA_SOURCE: ClassVar[DataSource] = DataSource.MTGTOP8
    DEFAULT_RAW_PATH: ClassVar[Path] = Path("data/raw/mtgtop8")

    def __init__(self) -> None:
        """
        Inputs: none. Output: none (constructor).
        Side effects: none. Exceptions: none.
        """
        super().__init__(MtgTop8DeckParser())

    def shard_rows(self, shards_dir: Path) -> Iterator[JsonObject]:
        """Every format directory's rows (read_shard_rows on each
        subdirectory of shards_dir, sorted by name; a format with no
        shards yet gives none).

        Inputs: shards_dir (data/raw/mtgtop8 or another crawl root).
        Output: a lazy iterator of rows.
        Side effects: lists shards_dir; reads the shard files.
        Exceptions: as read_shard_rows.

        Example:
            >>> next(MtgTop8DeckExtractionStage().shard_rows(Path("data/raw/mtgtop8")))
            {'deck_id': 909096, 'format': 'PAU', ...}
        """
        for format_dir in sorted(
            path for path in shards_dir.iterdir() if path.is_dir()
        ):
            yield from read_shard_rows(format_dir)
