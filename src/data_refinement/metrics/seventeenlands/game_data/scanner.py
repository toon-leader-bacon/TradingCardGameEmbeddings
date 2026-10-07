"""scan_game_csv() - one streaming read pass over a 17lands game_data
CSV across every Metric[GameDataChunk] at once (see game_data/README.md).

The scan itself is the shared ../chunk_scanner.py's scan_chunked_csv();
this module only fixes its chunk type to GameDataChunk.
"""

from pathlib import Path
from typing import Sequence

from src.data_refinement.metrics.metric import Metric
from src.data_refinement.metrics.seventeenlands.chunk_scanner import (
    DEFAULT_BLOCK_SIZE,
    scan_chunked_csv,
)
from src.data_refinement.seventeenlands.game_data.game_data_chunk import (
    GameDataChunk,
)
from src.data_refinement.seventeenlands.game_data.game_data_chunk_parser import (
    GameDataChunkParser,
)


def scan_game_csv(
    raw_csv_path: Path,
    metrics: Sequence[Metric[GameDataChunk]],
    parser: GameDataChunkParser,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> None:
    """Drive every metric over every chunk of raw_csv_path, then
    finalize all of them (scan_chunked_csv(), typed for game_data).

    Inputs: raw_csv_path (a game_data CSV), metrics, parser (built from
        this CSV's header), block_size (CSV bytes per record batch).
    Output: none.
    Side effects: as scan_chunked_csv().
    Exceptions: as scan_chunked_csv().

    Example:
        >>> parser = GameDataChunkParser.from_header(header, binder, GameId.MTG)
        >>> scan_game_csv(path, metrics, parser)
    """
    scan_chunked_csv(raw_csv_path, metrics, parser, block_size)
