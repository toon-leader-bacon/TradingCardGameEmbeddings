"""scan_draft_csv() - one streaming read pass over a 17lands draft_data
CSV across every Metric[DraftDataChunk] at once (see
draft_data/README.md).

The scan itself is the shared ../chunk_scanner.py's scan_chunked_csv();
this module only fixes its chunk type to DraftDataChunk.
"""

from pathlib import Path
from typing import Sequence

from src.data_refinement.metrics.metric import Metric
from src.data_refinement.metrics.seventeenlands.chunk_scanner import (
    DEFAULT_BLOCK_SIZE,
    scan_chunked_csv,
)
from src.data_refinement.metrics.seventeenlands.draft_data.draft_data_chunk import (
    DraftDataChunk,
)
from src.data_refinement.metrics.seventeenlands.draft_data.draft_data_chunk_parser import (
    DraftDataChunkParser,
)


def scan_draft_csv(
    raw_csv_path: Path,
    metrics: Sequence[Metric[DraftDataChunk]],
    parser: DraftDataChunkParser,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> None:
    """Drive every metric over every chunk of raw_csv_path, then
    finalize all of them (scan_chunked_csv(), typed for draft_data).

    Inputs: raw_csv_path (a draft_data CSV), metrics, parser (built from
        this CSV's header), block_size (CSV bytes per record batch).
    Output: none.
    Side effects: as scan_chunked_csv().
    Exceptions: as scan_chunked_csv().

    Example:
        >>> parser = DraftDataChunkParser.from_header(header, binder, GameId.MTG)
        >>> scan_draft_csv(path, metrics, parser)
    """
    scan_chunked_csv(raw_csv_path, metrics, parser, block_size)
