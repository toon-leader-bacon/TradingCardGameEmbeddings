"""FeatureRowKind: what the feature cache's stores need to know about
one feature type F, so one slab, one SQLite store, one chain and one
cache serve every cut.

Strategy. A store keeps rows of one width in storage units it calls
rows: a pooled feature is one row, a text's token states are one row
per token. A FeatureRowKind turns a batch of F into those rows (and
back), and selects and joins batches. Two kinds:

- POOLED_ROWS: F = Tensor (N, width), one row per text.
- TOKEN_ROWS: F = RaggedTokenStates, one row per real token.

Which kind a cache uses is not chosen here: it follows from the cache's
CutPoint (cut_point.py), the one source of truth for the cut.
"""

from typing import Protocol, Sequence, TypeVar

import torch

from src.encoder_model.feature_cache.card_text_key import CardTextKey
from src.encoder_model.feature_cache.ragged_token_states import RaggedTokenStates

F = TypeVar("F")


class FeatureRowKind(Protocol[F]):
    """Strategy: the row algebra of one feature type."""

    def text_count(self, features: F) -> int:
        """How many texts features holds. Inputs: features. Output: int.
        Side effects: none. Exceptions: none."""
        ...

    def row_counts(self, features: F) -> list[int]:
        """How many storage rows each text takes, in order (all 1 for
        pooled). Inputs: features. Output: list[int]. Side effects: none
        (token lengths already live on the host). Exceptions: none."""
        ...

    def flat_rows(self, features: F) -> torch.Tensor:
        """Every text's rows, one after another: (sum of row_counts,
        width), on features' device. Inputs: features. Output: Tensor.
        Side effects: none. Exceptions: none."""
        ...

    def from_flat_rows(self, rows: torch.Tensor, row_counts: Sequence[int]) -> F:
        """The inverse of flat_rows. Inputs: rows (sum(row_counts), width),
        row_counts (each >= 1). Output: F on rows' device and dtype.
        Side effects: none. Exceptions: ValueError if the counts do not
        sum to rows' length, or (pooled) a count is not 1."""
        ...

    def select(self, features: F, positions: Sequence[int]) -> F:
        """The given texts, in order (repeats allowed). Inputs: features,
        positions. Output: F. Side effects: none. Exceptions: IndexError
        for a position out of range."""
        ...

    def concatenate(self, parts: Sequence[F]) -> F:
        """parts' texts, one after another. Inputs: parts (non-empty, same
        width, device and dtype). Output: F. Side effects: none.
        Exceptions: ValueError for no parts."""
        ...

    def to(self, features: F, device: torch.device, dtype: torch.dtype) -> F:
        """features detached, moved and cast. The row kind owns the
        detach: stores and caches never keep a graph. Inputs: features,
        device, dtype. Output: F. Side effects: none. Exceptions: none."""
        ...


class PooledRowKind:
    """FeatureRowKind[Tensor]: (N, width) tensors, one row per text. Every
    operation is the plain tensor one, so the pooled path costs what it
    cost before row kinds existed."""

    def text_count(self, features: torch.Tensor) -> int:
        """Inputs: features (N, width). Output: N. Side effects: none.
        Exceptions: none."""
        return int(features.shape[0])

    def row_counts(self, features: torch.Tensor) -> list[int]:
        """Inputs: features (N, width). Output: [1] * N. Side effects:
        none. Exceptions: none."""
        return [1] * int(features.shape[0])

    def flat_rows(self, features: torch.Tensor) -> torch.Tensor:
        """Inputs: features. Output: features itself. Side effects: none.
        Exceptions: none."""
        return features

    def from_flat_rows(
        self, rows: torch.Tensor, row_counts: Sequence[int]
    ) -> torch.Tensor:
        """Inputs: rows (N, width), row_counts (N ones). Output: rows.
        Side effects: none. Exceptions: ValueError if a count is not 1 or
        there are not N of them."""
        if len(row_counts) != rows.shape[0] or any(n != 1 for n in row_counts):
            raise ValueError("a pooled feature is exactly one row per text")
        return rows

    def select(self, features: torch.Tensor, positions: Sequence[int]) -> torch.Tensor:
        """One index_select. Inputs: features, positions. Output: Tensor.
        Side effects: none. Exceptions: IndexError (from torch)."""
        index = torch.as_tensor(positions, dtype=torch.long, device=features.device)
        return features.index_select(0, index)

    def concatenate(self, parts: Sequence[torch.Tensor]) -> torch.Tensor:
        """torch.cat. Inputs: parts. Output: Tensor. Side effects: none.
        Exceptions: ValueError for no parts."""
        if not parts:
            raise ValueError("concatenate needs at least one part")
        return parts[0] if len(parts) == 1 else torch.cat(list(parts))

    def to(
        self, features: torch.Tensor, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Inputs: features, device, dtype. Output: detached Tensor.
        Side effects: none. Exceptions: none."""
        return features.detach().to(device, dtype)


class TokenRowKind:
    """FeatureRowKind[RaggedTokenStates]: one row per real token; every
    operation delegates to RaggedTokenStates."""

    def text_count(self, features: RaggedTokenStates) -> int:
        """Inputs: features. Output: features.text_count. Side effects:
        none. Exceptions: none."""
        return features.text_count

    def row_counts(self, features: RaggedTokenStates) -> list[int]:
        """Inputs: features. Output: features.lengths as a list.
        Side effects: none. Exceptions: none."""
        return [int(length) for length in features.lengths.tolist()]

    def flat_rows(self, features: RaggedTokenStates) -> torch.Tensor:
        """Inputs: features. Output: features.values. Side effects: none.
        Exceptions: none."""
        return features.values

    def from_flat_rows(
        self, rows: torch.Tensor, row_counts: Sequence[int]
    ) -> RaggedTokenStates:
        """Inputs: rows, row_counts. Output: RaggedTokenStates(rows,
        lengths). Side effects: none. Exceptions: ValueError (from
        RaggedTokenStates) if they disagree."""
        lengths = torch.tensor(list(row_counts), dtype=torch.int64)
        return RaggedTokenStates(rows, lengths)

    def select(
        self, features: RaggedTokenStates, positions: Sequence[int]
    ) -> RaggedTokenStates:
        """Inputs: features, positions. Output: features.select(positions).
        Side effects: none. Exceptions: IndexError."""
        return features.select(positions)

    def concatenate(self, parts: Sequence[RaggedTokenStates]) -> RaggedTokenStates:
        """Inputs: parts. Output: RaggedTokenStates.concatenate(parts).
        Side effects: none. Exceptions: ValueError for no parts."""
        return RaggedTokenStates.concatenate(parts)

    def to(
        self, features: RaggedTokenStates, device: torch.device, dtype: torch.dtype
    ) -> RaggedTokenStates:
        """Inputs: features, device, dtype. Output: RaggedTokenStates,
        values detached. Side effects: none. Exceptions: none."""
        detached = RaggedTokenStates(features.values.detach(), features.lengths)
        return detached.to(device, dtype)


# The two kinds hold no state: one instance of each serves every store
POOLED_ROWS = PooledRowKind()
TOKEN_ROWS = TokenRowKind()


def require_offer_shape(
    kind: FeatureRowKind[F],
    keys: Sequence[CardTextKey],
    features: F,
    width: int | None = None,
) -> list[int]:
    """The shape check every FeatureStore.put_many makes first, for any row
    kind.

    Inputs: kind, keys, features, width (the store's, or None while it
        has none).
    Output: list[int], the row count of each offered text (the check
        computes it anyway, and put_many needs it).
    Side effects: none.
    Exceptions: ValueError if the flat rows are not 2-D, the text count is
        not len(keys), keys repeat, or (with width) the width differs.

    Example:
        >>> require_offer_shape(POOLED_ROWS, [key_a], torch.zeros(1, 768), width=768)
        [1]
    """
    rows = kind.flat_rows(features)
    if rows.dim() != 2:
        raise ValueError(f"features must be 2-D rows, got shape {tuple(rows.shape)}")
    text_count = kind.text_count(features)
    if len(keys) != text_count:
        raise ValueError(f"{len(keys)} keys but {text_count} feature rows")
    if len(set(keys)) != len(keys):
        raise ValueError("put_many was offered the same key twice")
    if width is not None and rows.shape[1] != width:
        raise ValueError(f"features are {rows.shape[1]} wide, the store {width}")
    return kind.row_counts(features)
