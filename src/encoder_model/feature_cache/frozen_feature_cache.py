"""FrozenFeatureCache: what the frozen stages before a pipeline's cut
make of each card text, computed once and then looked up.

The feature type F follows the fingerprint's CutPoint: pooled vectors
(Tensor) for a pooled cut, RaggedTokenStates for a token cut. The cache
reaches F only through its store's FeatureRowKind, so every step below
is the same for both.

The one component CardEmbeddingPipeline talks to when it has a cache.
Keys are texts (CardTextKey), so a modded card is just another key; the
store (FeatureStore, usually a TieredFeatureStore) and its policies
decide what is kept. Every result is rounded to the store's dtype and
handed back as float32 on the caller's device, so a card's features
never depend on whether it hit or missed, or which tier answered.

The cache holds the FrozenStageFingerprint its rows were made under, and
refuses a call whose autocast differs from the fingerprint's compute
precision. Thread-safe: the store serializes itself, and the running
totals take their own lock; no lock is held while encoding.

A prefetch worker *reserves* a batch's texts: their keys are pinned in
every memory tier, then warmed, so no row the batch needs is evicted
before its step has looked it up. The worker releases the reservation
afterwards. Pinning before warming means a row is pinned from the moment
it is admitted.
"""

import threading
from dataclasses import dataclass
from typing import Callable, Generic, Sequence, TypeVar

import torch

from src.encoder_model.feature_cache.card_text_key import CardTextKey
from src.encoder_model.feature_cache.feature_row_kind import FeatureRowKind
from src.encoder_model.feature_cache.feature_store import (
    FeatureStore,
    StoreLookup,
    TierStats,
)
from src.encoder_model.feature_cache.frozen_stage_fingerprint import (
    FrozenStageFingerprint,
)
from src.encoder_model.precision import Precision

# The Precision each autocast dtype stands for (autocast off is fp32)
_PRECISION_OF_DTYPE: dict[torch.dtype, Precision] = {
    torch.float16: "fp16",
    torch.bfloat16: "bf16",
}

F = TypeVar("F")

# texts -> their features, live through the frozen stages before the cut
LiveFrozenStages = Callable[[list[str]], F]


@dataclass(frozen=True)
class FeatureCacheStats:
    """Running totals of one cache.

    texts: every text features() was asked for, duplicates included.
        Reservations are not counted: their texts are counted when the
        step asks for them.
    encoded: unique missing texts that went through the frozen stages,
        reservations' included.
    admitted: how many of those the store's top tier kept.
    tiers: each tier's hits and fill, top first.
    uncacheable: how many of the encoded texts were never offered to the
        store, because every card with that text had the do-not-cache
        stamp.

    The hit rate that matters for speed is 1 - encoded / texts: frozen
    encodes avoided, counting dedup within a call.
    """

    texts: int
    encoded: int
    admitted: int
    tiers: tuple[TierStats, ...]
    uncacheable: int = 0

    @property
    def avoided_fraction(self) -> float:
        """1 - encoded / texts; 0.0 before any text. Inputs: none.
        Output: float in [0, 1]. Side effects: none. Exceptions: none.

        Example:
            >>> FeatureCacheStats(10, 2, 2, ()).avoided_fraction
            0.8
        """
        if self.texts == 0:
            return 0.0
        return 1.0 - self.encoded / self.texts


@dataclass(frozen=True, eq=False)
class FeatureReservation:
    """Pins a cache holds for one reserved batch, until it is released.

    Compared by identity, not value: two reservations of the same texts
    are two sets of pins, each released once.

    keys: the batch's distinct keys; empty for a batch reserved without a
        cache (releasing it does nothing).
    """

    keys: tuple[CardTextKey, ...]


@dataclass(frozen=True)
class _UniqueTexts:
    """A batch of texts with duplicates folded.

    keys, texts: one entry per distinct text, in first-seen order.
    cacheable: per distinct text, whether any of its positions was
        cacheable (an unstamped card's text may be stored).
    inverse: for every input position, its index into keys.
    """

    keys: list[CardTextKey]
    texts: list[str]
    cacheable: list[bool]
    inverse: list[int]


class FrozenFeatureCache(Generic[F]):
    """Frozen-stage features by card text, over an injected FeatureStore
    (dependency injection: the storage and its policies are the store's)."""

    def __init__(
        self,
        store: FeatureStore[F],
        fingerprint: FrozenStageFingerprint,
        durable_misses: bool,
    ) -> None:
        """Inputs: store (empty or not; its storage dtype rounds every
            result), fingerprint (what the rows are made under; its compute
            precision is enforced), durable_misses (whether a miss encoded
            on the training path is kept by a durable tier).
        Output: none (constructor). Side effects: none.
        Exceptions: ValueError if the store's dtype is not the
            fingerprint's storage dtype, or its row kind is not the
            fingerprint's cut's.

        Example:
            >>> cache = FrozenFeatureCache(chain, fingerprint, durable_misses=False)
        """
        if store.storage_dtype != fingerprint.storage_dtype.torch_dtype:
            raise ValueError(
                f"the store holds {store.storage_dtype}, the fingerprint says "
                f"{fingerprint.storage_dtype.value}"
            )
        if store.row_kind is not fingerprint.cut.row_kind:
            raise ValueError(
                f"the store's row kind does not hold a {fingerprint.cut.describe()} cut"
            )
        self._store = store
        self._row_kind: FeatureRowKind[F] = store.row_kind
        self._fingerprint = fingerprint
        self._durable_misses = durable_misses
        self._totals_lock = threading.Lock()
        self._texts = 0
        self._encoded = 0
        self._admitted = 0
        self._uncacheable = 0
        # Reservations made and not yet released, by identity
        self._reservations_lock = threading.Lock()
        self._outstanding: set[FeatureReservation] = set()

    @property
    def fingerprint(self) -> FrozenStageFingerprint:
        """What this cache's rows are made under. Inputs: none.
        Output: FrozenStageFingerprint. Side effects: none. Exceptions: none."""
        return self._fingerprint

    def features(
        self,
        texts: Sequence[str],
        pool_live: LiveFrozenStages[F],
        device: torch.device,
        durable: bool | None = None,
        cacheable: Sequence[bool] | None = None,
    ) -> F:
        """Features for texts: stored ones looked up, the rest encoded by
        pool_live in one call and the cacheable ones offered to the store.

        Inputs: texts (non-empty, serialized cards; duplicates allowed),
            pool_live (the frozen stages: texts -> F on device),
            device (where the caller computes and the result goes), durable
            (whether the misses go to a durable tier; None: the cache's
            durable_misses), cacheable (parallel to texts: False for a
            stamped card's text, which is looked up and encoded but never
            offered; None: every text is cacheable).
        Output: detached float32 F on device, one entry per text in
            texts' order, each row rounded through the storage dtype.
        Side effects: runs pool_live under no_grad on the misses; the store
            may admit them and promote lower-tier hits; totals grow.
        Exceptions: ValueError if texts is empty, cacheable's length is not
            texts', or the active autocast on device's type is not the
            fingerprint's compute precision; whatever pool_live or the
            store raise.

        Example:
            >>> cache.features(["{...}", "{...}"], pipeline_pool_live, torch.device("cuda")).shape
            torch.Size([2, 768])
        """
        # Validate inputs
        if not texts:
            raise ValueError("features needs at least one text")
        flags = _cacheable_flags(texts, cacheable)
        self._require_precision(device)
        keep_durably = self._durable_misses if durable is None else durable
        with self._totals_lock:
            self._texts += len(texts)

        # Fold duplicates, then look up or encode each distinct text
        unique = _fold_duplicates(texts, flags)
        found, missed = self._look_up_or_encode(unique, pool_live, device, keep_durably)

        # Assemble the distinct rows, then expand back to texts' order
        result = self._assemble_distinct(found, missed, device)
        if len(unique.keys) < len(texts):
            result = self._row_kind.select(result, unique.inverse)
        return result

    def reserve(
        self,
        texts: Sequence[str],
        pool_live: LiveFrozenStages[F],
        device: torch.device,
        durable: bool | None = None,
        cacheable: Sequence[bool] | None = None,
    ) -> FeatureReservation:
        """Pin texts' keys in every memory tier, then warm them as features
        does, without returning rows. Until release, no tier evicts them.
        Uncacheable texts are skipped: no tier would keep them warm, so
        their encodes run in the step, and one that happens to be stored
        is not pinned.

        Inputs: texts (serialized cards; duplicates allowed; empty, or none
            cacheable, gives an empty reservation), pool_live, device,
            durable, cacheable (as features).
        Output: FeatureReservation over the distinct keys, outstanding
            until released.
        Side effects: store.pin; then features' side effects (lookup,
            promotion, encoding and offering misses), except that the
            texts total does not grow.
        Exceptions: ValueError if cacheable's length is not texts', or the
            active autocast is not the fingerprint's compute precision
            (before anything is pinned); whatever pool_live or the store
            raise, after unpinning.

        Example:
            >>> reservation = cache.reserve(batch_texts, pipeline_pool_live, torch.device("cuda"))
        """
        # Validate inputs
        flags = _cacheable_flags(texts, cacheable)
        kept = [text for text, flag in zip(texts, flags) if flag]
        if not kept:
            return FeatureReservation(())
        self._require_precision(device)
        keep_durably = self._durable_misses if durable is None else durable

        # Pin first, so every row is protected from the moment it lands
        unique = _fold_duplicates(kept, [True] * len(kept))
        self._store.pin(unique.keys)
        try:
            self._look_up_or_encode(unique, pool_live, device, keep_durably)
        except BaseException:
            self._store.unpin(unique.keys)
            raise

        result = FeatureReservation(tuple(unique.keys))
        with self._reservations_lock:
            self._outstanding.add(result)
        return result

    def release(self, reservation: FeatureReservation) -> None:
        """Remove a reservation's pins; its rows become evictable again.

        Inputs: reservation (made by this cache's reserve, not yet
            released; an empty one is always accepted).
        Output: none.
        Side effects: store.unpin of its keys; forgets the reservation.
        Exceptions: ValueError if reservation is not outstanding here
            (already released, or made by another cache).

        Example:
            >>> cache.release(reservation)
        """
        if not reservation.keys:
            return
        # Claim it first, so two releases cannot both unpin
        with self._reservations_lock:
            if reservation not in self._outstanding:
                raise ValueError("this reservation is not outstanding in this cache")
            self._outstanding.remove(reservation)
        self._store.unpin(reservation.keys)

    def stats(self) -> FeatureCacheStats:
        """The running totals and every tier's. Inputs: none.
        Output: FeatureCacheStats. Side effects: none. Exceptions: none.

        Example:
            >>> cache.stats().tiers[0].rows_used
            1261
        """
        with self._totals_lock:
            return FeatureCacheStats(
                self._texts,
                self._encoded,
                self._admitted,
                self._store.tier_stats(),
                self._uncacheable,
            )

    def close(self) -> None:
        """Close the store (its database connection, if any). Inputs: none.
        Output: none. Side effects: the store's close; the cache must not
        be used after. Exceptions: none.

        Example:
            >>> cache.close()
        """
        self._store.close()

    def _look_up_or_encode(
        self,
        unique: _UniqueTexts,
        pool_live: LiveFrozenStages[F],
        device: torch.device,
        durable: bool,
    ) -> tuple[StoreLookup[F], F | None]:
        """Look the distinct texts up, and encode and offer the misses.

        Private helper - consumers are features and reserve.
        Inputs: unique, pool_live, device, durable.
        Output: (the store's lookup, the encoded misses as float32 or None
            when nothing missed).
        Side effects: the store's lookup; _encode_misses on the misses.
        Exceptions: whatever pool_live or the store raise.
        """
        found = self._store.lookup(unique.keys, device)
        # Encode only the misses, round them as the store would, and offer
        # them; a rejected miss is still rounded the same way
        missed: F | None = None
        if found.miss_positions:
            missed = self._encode_misses(
                unique, found.miss_positions, pool_live, device, durable
            )
        return found, missed

    def _encode_misses(
        self,
        unique: _UniqueTexts,
        miss_positions: list[int],
        pool_live: LiveFrozenStages[F],
        device: torch.device,
        durable: bool,
    ) -> F:
        """Run pool_live on the missing texts, round to the storage dtype
        and offer the cacheable ones to the store.

        Private helper - single consumer is _look_up_or_encode.
        Inputs: unique, miss_positions (into unique), pool_live, device,
            durable.
        Output: float32 F (one per miss), detached, on device.
        Side effects: one no_grad pool_live call; store.put_many of the
            cacheable rows (none: no call); encoded, admitted and
            uncacheable totals grow.
        Exceptions: whatever pool_live or put_many raise.
        """
        keys = [unique.keys[position] for position in miss_positions]
        texts = [unique.texts[position] for position in miss_positions]
        # The frozen stages, without a graph: nothing before the cut trains
        kind = self._row_kind
        with torch.no_grad():
            encoded = pool_live(texts)
        rounded = kind.to(encoded, device, self._store.storage_dtype)
        # Offer only the cacheable rows; a stamped text is never stored
        offered = [
            row
            for row, position in enumerate(miss_positions)
            if unique.cacheable[position]
        ]
        admitted = 0
        if offered:
            offered_keys = [keys[row] for row in offered]
            offered_rows = rounded
            if len(offered) < len(keys):
                offered_rows = kind.select(rounded, offered)
            admitted = self._store.put_many(offered_keys, offered_rows, durable)
        with self._totals_lock:
            self._admitted += admitted
            self._encoded += len(keys)
            self._uncacheable += len(keys) - len(offered)
        return kind.to(rounded, device, torch.float32)

    def _assemble_distinct(
        self, found: StoreLookup[F], missed: F | None, device: torch.device
    ) -> F:
        """One float32 F with the hits and the encoded misses in their
        distinct-text positions.

        Private helper - single consumer is features.
        Inputs: found (hits in the storage dtype on device), missed
            (float32, or None when nothing missed; at least one of the
            two has rows), device.
        Output: float32 F on device. Side effects: none. Exceptions: none.
        """
        kind = self._row_kind
        # All hits or all misses: the rows are already in distinct order
        if missed is None:
            assert found.hits is not None  # every distinct text was a hit
            return kind.to(found.hits, device, torch.float32)
        if found.hits is None:
            return missed
        # Mixed: hits then misses, reordered by distinct position
        joined = kind.concatenate([kind.to(found.hits, device, torch.float32), missed])
        source_of = found.hit_positions + found.miss_positions
        order = sorted(range(len(source_of)), key=source_of.__getitem__)
        return kind.select(joined, order)

    def _require_precision(self, device: torch.device) -> None:
        """Private helper - consumers are features and reserve.

        Inputs: device (the caller's; its type names the autocast to read).
        Output: none. Side effects: none.
        Exceptions: ValueError if the active autocast ("fp32" when off) is
            not the fingerprint's compute precision.
        """
        active: Precision = "fp32"
        if torch.is_autocast_enabled(device.type):
            active = _PRECISION_OF_DTYPE.get(
                torch.get_autocast_dtype(device.type), "fp32"
            )
        expected = self._fingerprint.compute_precision
        if active != expected:
            raise ValueError(
                f"this feature cache was built for {expected} compute, but the call "
                f"runs at {active}; build the model with compute_precision={active!r}"
            )


def _cacheable_flags(
    texts: Sequence[str], cacheable: Sequence[bool] | None
) -> list[bool]:
    """cacheable as a list parallel to texts (all True for None).

    Private helper - consumers are FrozenFeatureCache.features and reserve.
    Inputs: texts, cacheable. Output: list[bool]. Side effects: none.
    Exceptions: ValueError if cacheable's length is not texts'.
    """
    if cacheable is None:
        return [True] * len(texts)
    if len(cacheable) != len(texts):
        raise ValueError(f"{len(cacheable)} cacheable flags for {len(texts)} texts")
    return list(cacheable)


def _fold_duplicates(texts: Sequence[str], cacheable: Sequence[bool]) -> _UniqueTexts:
    """Key every text once and record where each position's text went; a
    distinct text is cacheable if any of its positions is.

    Private helper - consumers are FrozenFeatureCache.features and reserve.
    Inputs: texts, cacheable (parallel to texts). Output: _UniqueTexts.
    Side effects: none. Exceptions: none.
    """
    result = _UniqueTexts(keys=[], texts=[], cacheable=[], inverse=[])
    # Fold on the text itself, so each distinct text is hashed only once
    index_of: dict[str, int] = {}
    for text, flag in zip(texts, cacheable):
        index = index_of.get(text)
        if index is None:
            index = len(result.texts)
            index_of[text] = index
            result.keys.append(CardTextKey.of_text(text))
            result.texts.append(text)
            result.cacheable.append(flag)
        elif flag:
            result.cacheable[index] = True
        result.inverse.append(index)
    return result

