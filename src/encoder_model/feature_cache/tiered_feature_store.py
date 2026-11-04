"""TieredFeatureStore: a chain of feature stores, fastest first.

Chain of Responsibility on lookup: each tier answers what it holds and
passes the rest down. The hierarchy is inclusive: every put is offered to
every tier (write-through), each tier's own policy decides what it keeps,
and a hit in a lower tier is offered to the tiers above it (promotion).
An eviction therefore needs no demotion copy: the row is usually already
below.
"""

from typing import Generic, Sequence, TypeVar

import torch

from src.encoder_model.feature_cache.card_text_key import CardTextKey
from src.encoder_model.feature_cache.feature_row_kind import FeatureRowKind
from src.encoder_model.feature_cache.feature_store import (
    FeatureStore,
    StoreLookup,
    TierStats,
)

F = TypeVar("F")


class TieredFeatureStore(Generic[F]):
    """FeatureStore[F] over an ordered chain of FeatureStore[F]."""

    def __init__(self, tiers: Sequence[FeatureStore[F]]) -> None:
        """Inputs: tiers, top (fastest) first.
        Output: none (constructor). Side effects: none.
        Exceptions: ValueError if tiers is empty, or their storage dtypes
            or row kinds differ.

        Example:
            >>> TieredFeatureStore([vram_slab, ram_slab, sqlite_store])
        """
        if not tiers:
            raise ValueError("a tiered store needs at least one tier")
        dtypes = {tier.storage_dtype for tier in tiers}
        if len(dtypes) > 1:
            raise ValueError(
                f"every tier must store one dtype, got {sorted(map(str, dtypes))}"
            )
        if len({id(tier.row_kind) for tier in tiers}) > 1:
            raise ValueError("every tier must hold the same row kind")
        self._tiers = list(tiers)
        self._row_kind: FeatureRowKind[F] = tiers[0].row_kind

    def lookup(
        self, keys: Sequence[CardTextKey], device: torch.device
    ) -> StoreLookup[F]:
        """Ask each tier in turn about the keys the tiers above missed;
        offer lower-tier hits to the tiers above.

        Inputs: keys (distinct), device.
        Output: StoreLookup over keys, hits on device.
        Side effects: every tier's own lookup side effects; promotion puts
            (not durable) into the tiers above each hit.
        Exceptions: whatever a tier raises.

        Example:
            >>> chain.lookup([in_ram, nowhere], torch.device("cuda")).miss_positions
            [1]
        """
        found_parts: list[tuple[list[int], F]] = []
        remaining = list(range(len(keys)))

        # Each tier answers the keys every tier above it missed
        for depth, tier in enumerate(self._tiers):
            if not remaining:
                break
            answer = tier.lookup([keys[p] for p in remaining], device)
            if answer.hits is not None:
                positions = [remaining[i] for i in answer.hit_positions]
                found_parts.append((positions, answer.hits))
                self._promote(depth, [keys[p] for p in positions], answer.hits)
            remaining = [remaining[i] for i in answer.miss_positions]

        result = _merge_parts(self._row_kind, found_parts, remaining)
        return result

    def put_many(self, keys: Sequence[CardTextKey], features: F, durable: bool) -> int:
        """Offer the rows to every tier (write-through).

        Inputs: keys (distinct), features, durable.
        Output: int, how many keys the top tier newly admitted.
        Side effects: every tier's put_many.
        Exceptions: whatever a tier raises.

        Example:
            >>> chain.put_many([key_a], torch.randn(1, 768, device="cuda"), durable=True)
            1
        """
        admitted = [tier.put_many(keys, features, durable) for tier in self._tiers]
        return admitted[0]

    def pin(self, keys: Sequence[CardTextKey]) -> None:
        """Pin keys in every tier, top first.

        Inputs: keys (distinct). Output: none.
        Side effects: every tier's pin. Exceptions: whatever a tier raises.

        Example:
            >>> chain.pin([key_a])
        """
        for tier in self._tiers:
            tier.pin(keys)

    def unpin(self, keys: Sequence[CardTextKey]) -> None:
        """Unpin keys in every tier, top first. Each tier is all or nothing;
        the chain is not (a tier that raises leaves the tiers above
        unpinned), which only a caller bug can trigger: the cache unpins
        exactly what it pinned.

        Inputs: keys (distinct, pinned). Output: none.
        Side effects: every tier's unpin.
        Exceptions: ValueError from a tier with no pin for a key.

        Example:
            >>> chain.unpin([key_a])
        """
        for tier in self._tiers:
            tier.unpin(keys)

    def tier_stats(self) -> tuple[TierStats, ...]:
        """Every tier's stats, top first. Inputs: none. Output: tuple.
        Side effects: none. Exceptions: none."""
        return tuple(stats for tier in self._tiers for stats in tier.tier_stats())

    @property
    def storage_dtype(self) -> torch.dtype:
        """The tiers' shared dtype. Inputs: none. Output: torch.dtype.
        Side effects: none. Exceptions: none."""
        return self._tiers[0].storage_dtype

    @property
    def row_kind(self) -> FeatureRowKind[F]:
        """The tiers' shared row kind. Inputs: none. Output: FeatureRowKind.
        Side effects: none. Exceptions: none."""
        return self._row_kind

    def close(self) -> None:
        """Close every tier. Inputs: none. Output: none.
        Side effects: every tier's close. Exceptions: none."""
        for tier in self._tiers:
            tier.close()

    def _promote(self, depth: int, keys: list[CardTextKey], features: F) -> None:
        """Offer rows found at depth to every tier above it (not durably).

        Private helper - single consumer is lookup.
        Inputs: depth (0: nothing above), keys, features (on the asked
            device). Output: none. Side effects: put_many on those tiers.
        Exceptions: whatever a tier raises.
        """
        for tier in self._tiers[:depth]:
            tier.put_many(keys, features, durable=False)


def _merge_parts(
    kind: FeatureRowKind[F],
    found_parts: list[tuple[list[int], F]],
    miss_positions: list[int],
) -> StoreLookup[F]:
    """One StoreLookup from per-tier hits, with hit_positions ascending and
    hits reordered to match.

    Private helper - single consumer is TieredFeatureStore.lookup.
    Inputs: kind (the tiers' row kind), found_parts ((positions, hits)
        per tier that hit),
        miss_positions (ascending). Output: StoreLookup.
    Side effects: none. Exceptions: none.
    """
    if not found_parts:
        return StoreLookup([], None, miss_positions)
    if len(found_parts) == 1:
        positions, rows = found_parts[0]
        return StoreLookup(positions, rows, miss_positions)
    # Several tiers hit: concatenate, then sort rows by position
    positions = [p for part_positions, _ in found_parts for p in part_positions]
    hits = kind.concatenate([part_hits for _, part_hits in found_parts])
    order = sorted(range(len(positions)), key=positions.__getitem__)
    return StoreLookup(sorted(positions), kind.select(hits, order), miss_positions)
