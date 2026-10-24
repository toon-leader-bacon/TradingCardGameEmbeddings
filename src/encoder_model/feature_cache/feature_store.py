"""FeatureStore: the storage boundary of the feature cache.

A store maps CardTextKeys to feature rows, batched both ways, and
honours pins: a pinned key's row is never evicted from a memory tier.
Which technology holds the rows (a VRAM or pinned-RAM slab, a SQLite file, or a
chain of them) and which keys it keeps (its SlotPolicy) are the store's
business; the cache only looks up and offers. Every store is
thread-safe: a prefetch worker and the training thread may call it at
once.

Generic in the feature type F: (N, dim) tensors at the pooled cut; a
ragged token-feature type at a token-level cut later.
"""

from dataclasses import dataclass
from typing import Generic, Protocol, Sequence, TypeVar

import torch

from src.encoder_model.feature_cache.card_text_key import CardTextKey

F = TypeVar("F")


@dataclass(frozen=True)
class StoreLookup(Generic[F]):
    """One batched lookup's answer.

    hit_positions: indices into the queried keys that the store holds,
        ascending.
    hits: their features, in hit_positions' order, on the device asked
        for; None exactly when hit_positions is empty.
    miss_positions: every other index into the queried keys, ascending.
    """

    hit_positions: list[int]
    hits: F | None
    miss_positions: list[int]

    def __post_init__(self) -> None:
        """Inputs: none (checks self). Output: none. Side effects: none.
        Exceptions: ValueError if hits is None while hit_positions is
        non-empty, or not None while it is empty."""
        if (self.hits is None) != (not self.hit_positions):
            raise ValueError("hits must be None exactly when there are no hits")


@dataclass(frozen=True)
class TierStats:
    """One storage tier's running totals.

    name: "vram", "ram" or "disk".
    hits: keys this tier answered (a chain asks a lower tier only about
        the upper tiers' misses).
    rows_used, capacity_rows: its fill.
    pinned_rows: held rows with at least one pin (0 for the disk).
    refused_for_pins: running total of offered keys the tier's policy
        would have admitted but for pinned rows filling it.
    evicted: running total of rows dropped to make room (0 for a tier
        that never evicts).
    """

    name: str
    hits: int
    rows_used: int
    capacity_rows: int
    pinned_rows: int
    refused_for_pins: int
    evicted: int = 0


class FeatureStore(Protocol[F]):
    """Batched, thread-safe key -> feature storage with its own admission."""

    def lookup(
        self, keys: Sequence[CardTextKey], device: torch.device
    ) -> StoreLookup[F]:
        """Inputs: keys (distinct), device (where hits should land).
        Output: StoreLookup over those positions, hits on device.
        Side effects: tells the store's policy which keys were hit; a chain
            offers lower-tier hits to the tiers above.
        Exceptions: none of its own."""
        ...

    def put_many(self, keys: Sequence[CardTextKey], features: F, durable: bool) -> int:
        """Offer features for keys; the store's policy decides which are
        kept. Keys already held are skipped, not an error (two threads may
        encode the same miss).
        Inputs: keys (distinct), features (one row per key, same order, on
            any device: each tier copies to its own placement), durable
            (whether a persistent tier should keep them).
        Output: int, how many keys this store newly admitted (a chain: its
            top tier).
        Side effects: may write rows and evict others.
        Exceptions: ValueError for a shape or width mismatch (see
            require_offer_shape); RuntimeError from a slab whose policy
            decided something malformed; sqlite3.Error from a disk tier."""
        ...

    def pin(self, keys: Sequence[CardTextKey]) -> None:
        """Add one pin to each key, held or not yet held: a memory tier
        never evicts a key while it has a pin. Pins are reference counts,
        since several reservations can share a key. A chain pins every
        tier; the disk tier does nothing.
        Inputs: keys (distinct). Output: none.
        Side effects: increments the keys' pin counts.
        Exceptions: none of its own."""
        ...

    def unpin(self, keys: Sequence[CardTextKey]) -> None:
        """Remove one pin from each key. All or nothing within a tier: the
        counts are checked before any changes. The disk tier does nothing.
        Inputs: keys (distinct, each pinned). Output: none.
        Side effects: decrements the keys' pin counts; a key at zero is
            evictable again.
        Exceptions: ValueError if a key has no pin in a memory tier."""
        ...

    def tier_stats(self) -> tuple[TierStats, ...]:
        """Each tier's totals, top first (one entry for a single store).
        Inputs: none. Output: tuple[TierStats, ...]. Side effects: none.
        Exceptions: none."""
        ...

    @property
    def storage_dtype(self) -> torch.dtype:
        """The dtype rows are stored in. Inputs: none. Output: torch.dtype.
        Side effects: none. Exceptions: none."""
        ...

    def close(self) -> None:
        """Release what the store holds open (a database connection);
        nothing for a memory tier. Inputs: none. Output: none.
        Side effects: as described; the store must not be used after.
        Exceptions: none."""
        ...


def require_offer_shape(
    keys: Sequence[CardTextKey], features: torch.Tensor, width: int | None = None
) -> None:
    """The shape check every FeatureStore.put_many makes first.

    Inputs: keys, features, width (the store's, or None while it has none).
    Output: none. Side effects: none.
    Exceptions: ValueError if features is not 2-D, its row count is not
        len(keys), keys repeat, or (with width) its width differs.

    Example:
        >>> require_offer_shape([key_a], torch.zeros(1, 768), width=768)
    """
    if features.dim() != 2:
        raise ValueError(f"features must be 2-D, got shape {tuple(features.shape)}")
    if len(keys) != features.shape[0]:
        raise ValueError(f"{len(keys)} keys but {features.shape[0]} feature rows")
    if len(set(keys)) != len(keys):
        raise ValueError("put_many was offered the same key twice")
    if width is not None and features.shape[1] != width:
        raise ValueError(f"features are {features.shape[1]} wide, the store {width}")
