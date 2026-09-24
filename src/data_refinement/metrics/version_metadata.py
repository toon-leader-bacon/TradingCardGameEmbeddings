"""Embeds/reads which CardBinder snapshot a metric's output parquet
file was built from, plus whether a DeckBox is needed to fully verify
it.

See plans/card_binder_versioning.md for the full design. Stored as
native parquet schema metadata - part of the one output file's own
footer, not a sidecar - so it can never be separated from, or
mismatched with, the file it describes. Every metric that writes via
pyarrow.parquet.ParquetWriter already builds its output pa.Schema once
in __init__, before opening the writer; schema_with_version_metadata()
is that schema construction's last step. A metric writing via
pandas.DataFrame.to_parquet() instead uses
write_dataframe_with_version_metadata(), which reaches the same on-disk
result through pyarrow directly.

NO DECK-BOX CONTENT HASH HERE, DELIBERATELY: an earlier version of this
module carried a `deck_box_version` (a DeckBox.version_for()-shaped
content hash) alongside card_binder_version. That doesn't work for any
metric that populates its own DeckBox as a side effect of the very
scan that writes this file (every DeckBox-consuming metric in this
project does exactly that - see e.g. deck_card_mask_metric.py,
sts_gg/deck_label_metric.py): a ParquetWriter's schema (metadata
included) is fixed the moment it opens, before a single row is
written, so there is no point in __init__ at which "the DeckBox's
final, post-scan content" is already known to hash. `requires_deck_box`
below is the correct, cheaply-computable signal instead: it says
"verify this metric's card_binder_version against whatever CardBinder
the given DeckBox itself claims to have been minted from"
(DeckBox.card_binder_version_for()) rather than trying to hash the
DeckBox's content at a time that content isn't finalized yet. A dojo
reading from an independently-published, already-final DeckBox (e.g.
ContrastiveDojo/DeckBoxDealer) doesn't go through this module at all -
see DeckBoxDealer.card_binder_version for that case.
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.schema.game_id import GameId

_GAME_KEY = b"game"
_CARD_BINDER_VERSION_KEY = b"card_binder_version"
_REQUIRES_DECK_BOX_KEY = b"requires_deck_box"
_TRUE_BYTES = b"1"
_FALSE_BYTES = b"0"


@dataclass(frozen=True)
class MetricVersionMetadata:
    """Which CardBinder snapshot a metric output was built from, and
    whether checking it fully also requires a DeckBox.

    Inputs: none (data holder).
    Output: n/a.
    Side effects: none.
    Exceptions: none.
    """

    game: GameId
    card_binder_version: str
    requires_deck_box: bool = False


def schema_with_version_metadata(
    schema: pa.Schema, metadata: MetricVersionMetadata
) -> pa.Schema:
    """Attach metadata to schema as bytes-valued parquet schema metadata.

    Inputs:
        schema: an output schema a metric is about to open a
            ParquetWriter with.
        metadata: this metric's source version(s).
    Output: schema, with metadata's fields merged into its existing
        metadata (if any) under b"game"/b"card_binder_version"/
        b"requires_deck_box".
    Side effects: none - schema itself is immutable; this returns a
        new pa.Schema.
    Exceptions: none.

    Example:
        >>> schema = schema_with_version_metadata(
        ...     pa.schema([("nocab_uuid", pa.string())]),
        ...     MetricVersionMetadata(GameId.GWENT, "3f2b1c..."),
        ... )
    """
    existing = schema.metadata or {}
    new_metadata = {
        **existing,
        _GAME_KEY: metadata.game.value.encode("utf-8"),
        _CARD_BINDER_VERSION_KEY: metadata.card_binder_version.encode("utf-8"),
        _REQUIRES_DECK_BOX_KEY: (
            _TRUE_BYTES if metadata.requires_deck_box else _FALSE_BYTES
        ),
    }
    return schema.with_metadata(new_metadata)


def write_dataframe_with_version_metadata(
    df: pd.DataFrame, path: Path, metadata: MetricVersionMetadata
) -> None:
    """Write df to path as parquet, with metadata embedded.

    For the pandas.DataFrame.to_parquet()-based metric writers (as
    opposed to the ParquetWriter-based ones, which call
    schema_with_version_metadata() directly on their own schema).

    Inputs:
        df: the metric's complete output rows.
        path: destination file. Overwritten if it already exists.
        metadata: this metric's source version(s).
    Output: none.
    Side effects: creates path's parent directories if missing; writes
        path.
    Exceptions: whatever pyarrow.parquet.write_table raises.

    Example:
        >>> write_dataframe_with_version_metadata(
        ...     df, Path("data/metrics/dominiontabs/set_mask.parquet"),
        ...     MetricVersionMetadata(GameId.DOMINION, "9c1a2f..."),
        ... )
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    table = table.replace_schema_metadata(
        schema_with_version_metadata(table.schema, metadata).metadata
    )
    pq.write_table(table, path)


def metadata_from_schema(schema: pa.Schema) -> MetricVersionMetadata | None:
    """Parse an already-in-hand pa.Schema's metadata back into a
    MetricVersionMetadata.

    Pure, no I/O - the one place this project's byte-key parsing rule
    is written; both read_version_metadata() below and a dojo holding
    a schema it already opened (e.g. via FileManagerParquet.schema)
    call this rather than duplicating it.

    Inputs:
        schema: a schema read from an on-disk parquet file (or one
            built in-process via schema_with_version_metadata()).
    Output: the embedded MetricVersionMetadata, or None if schema
        carries no b"game"/b"card_binder_version" keys (a file that
        predates this module, or was never version-stamped).
        requires_deck_box defaults to False if the key is absent (a
        file written before that field existed).
    Side effects: none.
    Exceptions: none.

    Example:
        >>> metadata_from_schema(pq.ParquetFile(path).schema_arrow)
    """
    raw_metadata = schema.metadata or {}
    if _GAME_KEY not in raw_metadata or _CARD_BINDER_VERSION_KEY not in raw_metadata:
        return None
    return MetricVersionMetadata(
        game=GameId(raw_metadata[_GAME_KEY].decode("utf-8")),
        card_binder_version=raw_metadata[_CARD_BINDER_VERSION_KEY].decode("utf-8"),
        requires_deck_box=raw_metadata.get(_REQUIRES_DECK_BOX_KEY) == _TRUE_BYTES,
    )


def read_version_metadata(path: Path) -> MetricVersionMetadata | None:
    """Read path's embedded MetricVersionMetadata, opening it fresh.

    For a caller that only has a path, not an already-open schema
    (e.g. a standalone inventory/inspection script). GenericDojo does
    NOT call this - it already has a schema in hand via
    FileManagerParquet.schema and calls metadata_from_schema()
    directly, so its version check never re-opens a file
    FileManagerParquet.__init__ already opened.

    Inputs:
        path: an on-disk parquet file.
    Output: see metadata_from_schema().
    Side effects: opens path to read its schema (no row data read).
    Exceptions: whatever pyarrow.parquet.ParquetFile raises for a
        missing/corrupt file.

    Example:
        >>> read_version_metadata(Path("data/metrics/gwent_one/color_mask.parquet"))
    """
    return metadata_from_schema(pq.ParquetFile(path).schema_arrow)
