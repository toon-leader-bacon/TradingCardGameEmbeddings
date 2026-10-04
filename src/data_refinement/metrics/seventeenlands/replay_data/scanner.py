"""scan_replay_csv() - one streaming read pass over a 17lands
replay_data CSV across every Metric[ReplayDataChunk] at once (see
replay_data/README.md).

The scan itself is the shared ../chunk_scanner.py's scan_chunked_csv();
this module only fixes its chunk type to ReplayDataChunk.
"""

from pathlib import Path
from typing import Sequence

from src.data_refinement.metrics.metric import Metric
from src.data_refinement.metrics.seventeenlands.chunk_scanner import (
    DEFAULT_BLOCK_SIZE,
    scan_chunked_csv,
)
from src.data_refinement.metrics.seventeenlands.replay_data.replay_data_chunk import (
    ReplayDataChunk,
)
from src.data_refinement.metrics.seventeenlands.replay_data.replay_data_chunk_parser import (
    ReplayDataChunkParser,
)


def scan_replay_csv(
    raw_csv_path: Path,
    metrics: Sequence[Metric[ReplayDataChunk]],
    parser: ReplayDataChunkParser,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> None:
    """Drive every metric over every chunk of raw_csv_path, then
    finalize all of them (scan_chunked_csv(), typed for replay_data).

    Inputs: raw_csv_path (a replay_data CSV), metrics, parser (built
        from this CSV's header), block_size (CSV bytes per record batch).
    Output: none.
    Side effects: as scan_chunked_csv().
    Exceptions: as scan_chunked_csv().

    Example:
        >>> parser = ReplayDataChunkParser.from_header(header, binder, GameId.MTG)
        >>> scan_replay_csv(path, metrics, parser)
    """
    scan_chunked_csv(raw_csv_path, metrics, parser, block_size)
