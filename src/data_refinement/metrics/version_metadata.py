"""Embeds/reads which CardBinder snapshot a metric's output parquet
file was built from, plus whether a DeckBox is needed to fully verify
it.

See metrics/README.md for how the version check is used. Stored as
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
import json
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.schema.game_id import GameId

_GAME_KEY = b"game"
_CARD_BINDER_VERSION_KEY = b"card_binder_version"
_REQUIRES_DECK_BOX_KEY = b"requires_deck_box"
_CARD_BINDER_VERSIONS_KEY = b"card_binder_versions"
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


@dataclass(frozen=True)
class MultiGameVersionMetadata:
    """Which CardBinder snapshot of each game a multi-game metric output
    was built from (a cross-game metric, whose one file spans games).

    card_binder_versions: game -> CardLookup.version_for(game), for every
        game the file has rows from.

    Stored under its own schema key, so a file carrying it and a
    single-game MetricVersionMetadata file never confuse each other.

    Inputs: none (data holder). Output: n/a. Side effects: none.
    Exceptions: none.
    """

    card_binder_versions: Mapping[GameId, str]

    def __post_init__(self) -> None:
        """Copy the mapping into a read-only view, so a caller's later
        edits to its own dict cannot change this record."""
        object.__setattr__(
            self,
            "card_binder_versions",
            MappingProxyType(dict(self.card_binder_versions)),
        )


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
    _write_dataframe_with_stamp(
        df, path, lambda schema: schema_with_version_metadata(schema, metadata)
    )


def _write_dataframe_with_stamp(
    df: pd.DataFrame, path: Path, stamp: Callable[[pa.Schema], pa.Schema]
) -> None:
    """Write df to path as parquet, its footer metadata taken from
    stamp(df's schema). The one writer behind both public ones.

    Inputs: df, path (overwritten), stamp (adds the version metadata to a
        schema, e.g. schema_with_version_metadata bound to its metadata).
    Output: none.
    Side effects: creates path's parent directories; writes path.
    Exceptions: whatever pyarrow.parquet.write_table raises.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    table = table.replace_schema_metadata(stamp(table.schema).metadata)
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


def schema_with_multi_game_versions(
    schema: pa.Schema, metadata: MultiGameVersionMetadata
) -> pa.Schema:
    """Attach per-game binder versions to schema as parquet schema
    metadata (one JSON object, game value -> version).

    Inputs: schema (an output schema), metadata.
    Output: a new pa.Schema; existing metadata is kept.
    Side effects: none.
    Exceptions: none.

    Example:
        >>> schema_with_multi_game_versions(
        ...     pa.schema([("nocab_uuid", pa.string())]),
        ...     MultiGameVersionMetadata({GameId.GWENT: "3f2b1c..."}),
        ... )
    """
    versions = {
        game.value: version for game, version in metadata.card_binder_versions.items()
    }
    stamp = json.dumps(versions, sort_keys=True).encode("utf-8")
    return schema.with_metadata(
        {**(schema.metadata or {}), _CARD_BINDER_VERSIONS_KEY: stamp}
    )


def write_dataframe_with_multi_game_versions(
    df: pd.DataFrame, path: Path, metadata: MultiGameVersionMetadata
) -> None:
    """Write df to path as parquet with metadata embedded - the
    multi-game twin of write_dataframe_with_version_metadata().

    Inputs: df (the metric's complete rows), path (overwritten if it
        exists), metadata.
    Output: none.
    Side effects: creates path's parent directories; writes path.
    Exceptions: whatever pyarrow.parquet.write_table raises.

    Example:
        >>> write_dataframe_with_multi_game_versions(
        ...     df, Path("data/metrics/cross_game/rarity_tier.parquet"),
        ...     MultiGameVersionMetadata({GameId.MTG: "9c1a2f..."}),
        ... )
    """
    _write_dataframe_with_stamp(
        df, path, lambda schema: schema_with_multi_game_versions(schema, metadata)
    )


def multi_game_versions_from_schema(
    schema: pa.Schema,
) -> MultiGameVersionMetadata | None:
    """Parse a schema's per-game binder versions back out.

    Inputs: schema (read from an on-disk parquet file).
    Output: the embedded MultiGameVersionMetadata, or None if the schema
        carries no per-game versions (every single-game file).
    Side effects: none.
    Exceptions: ValueError if the stored value is not a JSON object of
        known game values to strings.

    Example:
        >>> multi_game_versions_from_schema(
        ...     pq.ParquetFile(path).schema_arrow)
    """
    raw_metadata = schema.metadata or {}
    if _CARD_BINDER_VERSIONS_KEY not in raw_metadata:
        return None

    # Parse the stored JSON object into known games and string versions
    try:
        stored = json.loads(raw_metadata[_CARD_BINDER_VERSIONS_KEY].decode("utf-8"))
        if not isinstance(stored, dict) or not all(
            isinstance(version, str) for version in stored.values()
        ):
            raise ValueError("not an object of strings")
        return MultiGameVersionMetadata(
            {GameId(game): version for game, version in stored.items()}
        )
    except ValueError as error:
        raise ValueError(f"bad stored card_binder_versions: {error}") from error
