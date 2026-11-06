"""EmbeddingTableMetadata: what produced an EmbeddingTable, and its lossless
JSON form (stored in the table's table_info row)."""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, TypeVar

from src.schema.game_id import GameId

_Value = TypeVar("_Value")


@dataclass(frozen=True)
class EmbeddingTableMetadata:
    """What produced a table. Recorded once at EmbeddingTable.create(), never checked
    afterwards (extending a table from a different binder version is the
    caller's responsibility).

    encoder_label: name used in output paths and plots (non-empty).
    checkpoint_dir: where the encoder came from; None for an untrained one.
    embedding_dim: width of every stored vector (>= 1).
    binder_versions: CardLookup.version_for per game. Its keys are the
        table's games, fixed at EmbeddingTable.create(): EmbeddingTable.add() rejects any
        other game.
        Stored as a read-only copy (MappingProxyType), so mutating the
        caller's dict afterwards cannot change the table's games.
    created_at: when the table was created.

    Exceptions: ValueError on construction if encoder_label is empty,
        embedding_dim < 1, or binder_versions is empty.

    Not hashable (binder_versions is a mapping); compare with ==.
    """

    encoder_label: str
    checkpoint_dir: Path | None
    embedding_dim: int
    binder_versions: Mapping[GameId, str]
    created_at: datetime

    def __post_init__(self) -> None:
        # Validate the fields, then freeze binder_versions as a read-only
        # copy (object.__setattr__: the dataclass is frozen)
        if not self.encoder_label:
            raise ValueError("encoder_label must be non-empty")
        if self.embedding_dim < 1:
            raise ValueError(f"embedding_dim must be >= 1, got {self.embedding_dim}")
        if not self.binder_versions:
            raise ValueError("binder_versions needs at least one game")
        frozen = MappingProxyType(dict(self.binder_versions))
        object.__setattr__(self, "binder_versions", frozen)

    @property
    def games(self) -> frozenset[GameId]:
        """The games this table may hold (binder_versions' keys).
        Inputs: none. Output: frozenset[GameId]. Side effects: none.
        Exceptions: none."""
        return frozenset(self.binder_versions)

    def to_json(self) -> str:
        """Serialize losslessly: from_json(metadata.to_json()) == metadata.

        Inputs: none.
        Output: str, JSON with keys sorted; paths and datetimes as strings
            (ISO 8601), games by GameId value.
        Side effects: none. Exceptions: none.

        Example:
            >>> EmbeddingTableMetadata.from_json(metadata.to_json()) == metadata
            True
        """
        raw = {
            "encoder_label": self.encoder_label,
            "checkpoint_dir": (
                None if self.checkpoint_dir is None else str(self.checkpoint_dir)
            ),
            "embedding_dim": self.embedding_dim,
            "binder_versions": {
                game.value: version for game, version in self.binder_versions.items()
            },
            "created_at": self.created_at.isoformat(),
        }
        return json.dumps(raw, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "EmbeddingTableMetadata":
        """Parse text written by to_json.

        Inputs: text (str).
        Output: the equal EmbeddingTableMetadata (same frozen form:
            binder_versions read-only).
        Side effects: none.
        Exceptions: ValueError on malformed JSON, a missing key, a wrong
            type, an unknown GameId, a bad timestamp, or invalid fields.

        Example:
            >>> EmbeddingTableMetadata.from_json(text).encoder_label
            'single_v1'
        """
        try:
            raw = json.loads(text)
            checkpoint_dir = raw["checkpoint_dir"]
            versions = _require_json_type(
                raw["binder_versions"], dict, "binder_versions"
            )
            return cls(
                encoder_label=_require_json_type(
                    raw["encoder_label"], str, "encoder_label"
                ),
                checkpoint_dir=(
                    None
                    if checkpoint_dir is None
                    else Path(_require_json_type(checkpoint_dir, str, "checkpoint_dir"))
                ),
                embedding_dim=_require_json_type(
                    raw["embedding_dim"], int, "embedding_dim"
                ),
                binder_versions={
                    GameId(game): _require_json_type(version, str, "binder version")
                    for game, version in versions.items()
                },
                created_at=datetime.fromisoformat(
                    _require_json_type(raw["created_at"], str, "created_at")
                ),
            )
        except (KeyError, TypeError) as error:
            # A missing key, or a JSON value that is not an object
            raise ValueError(f"malformed metadata: {error!r}") from error


def _require_json_type(value: object, kind: type[_Value], name: str) -> _Value:
    """Inputs: a parsed JSON value, the type it must have, its field name.
    Output: value, typed. Side effects: none. Exceptions: ValueError
    naming the field if value is not a kind (a bool is never an int)."""
    is_bool_posing_as_int = isinstance(value, bool) and kind is not bool
    if is_bool_posing_as_int or not isinstance(value, kind):
        raise ValueError(f"{name} must be {kind.__name__}, got {value!r}")
    return value
