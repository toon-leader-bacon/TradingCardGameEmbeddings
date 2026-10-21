"""TensorFeatureSlab: a FeatureStore[Tensor] holding fixed-width feature
rows in one preallocated tensor, in VRAM or in pinned host memory.

The rows live in one (capacity, width) tensor, placed by SlabPlacement:
on the device of the first put (VRAM in training), or in pinned CPU
memory (the RAM tier). Keys and the key -> row index stay in a CPU dict,
so a lookup reads nothing back from the device: one dict pass and one
gather. The slab is not an nn.Module, so model.to(device) does not move
it and it is never in a state_dict.

Thread safety: a lock serializes lookup and put_many, and on CUDA every
gather and write runs on the device's default stream, synced with the
caller's stream on the way in and out. The lock orders the enqueues; the
single stream makes the GPU run them in that order, so a gather never
sees a row half-written or already overwritten by a later eviction. A
hits tensor handed to a caller on another stream is recorded on that
stream (record_stream), so the allocator cannot reuse its memory while
the caller still reads it. A pinned slab's rows are written with a
blocking copy: its gathers are CPU work that no stream orders.

Pins (for prefetch reservations) depend on that ordering. A reservation
is released once its step's lookup has gathered its own copy of the
rows; a later eviction may then overwrite those rows, and only this
slab's single stream (or, pinned, the synchronous gather) guarantees the
overwrite runs after the gather. A change to the slab's streams must
keep this.
"""

import contextlib
import threading
from collections import Counter
from enum import Enum
from typing import Iterator, Sequence

import torch

from src.encoder_model.feature_cache.card_text_key import CardTextKey
from src.encoder_model.feature_cache.feature_storage_dtype import FeatureStorageDtype
from src.encoder_model.feature_cache.feature_store import (
    StoreLookup,
    TierStats,
    require_offer_shape,
)
from src.encoder_model.feature_cache.slot_policy import AdmissionDecision, SlotPolicy


class SlabPlacement(Enum):
    """Where a slab keeps its rows."""

    FOLLOW_FIRST_PUT = "follow_first_put"  # the first put's device (VRAM)
    PINNED_HOST = "pinned_host"  # pinned CPU memory (the RAM tier)


class TensorFeatureSlab:
    """FeatureStore[Tensor]: fixed-width rows in one lazily allocated
    tensor, admission and eviction by an injected SlotPolicy (Strategy)."""

    def __init__(
        self,
        name: str,
        capacity_rows: int,
        storage_dtype: FeatureStorageDtype,
        policy: SlotPolicy,
        placement: SlabPlacement,
    ) -> None:
        """Inputs: name (for stats, e.g. "vram"), capacity_rows (>= 1),
            storage_dtype (rows are stored in it), policy (decides
            admission and eviction; sized for capacity_rows), placement.
        Output: none (constructor). Side effects: none; nothing is
            allocated until the first put_many.
        Exceptions: ValueError if capacity_rows < 1.

        Example:
            >>> slab = TensorFeatureSlab("ram", 1000, FeatureStorageDtype.FP16,
            ...                          LeastRecentlyUsed(1000), SlabPlacement.PINNED_HOST)
        """
        if capacity_rows < 1:
            raise ValueError(f"capacity_rows must be >= 1, got {capacity_rows}")
        self._name = name
        self._capacity_rows = capacity_rows
        self._storage_dtype = storage_dtype.torch_dtype
        self._policy = policy
        self._placement = placement
        self._lock = threading.Lock()
        self._row_of: dict[CardTextKey, int] = {}
        # Rows freed by eviction, reused before any never-used row
        self._released_rows: list[int] = []
        self._next_unused_row = 0
        self._hits = 0
        # Pin counts by key, held or not yet held; a key leaves at zero
        self._pins: Counter[CardTextKey] = Counter()
        self._refused_for_pins = 0
        # (capacity_rows, width) once the first put fixes device and width
        self._rows: torch.Tensor | None = None

    def lookup(
        self, keys: Sequence[CardTextKey], device: torch.device
    ) -> StoreLookup[torch.Tensor]:
        """Which keys the slab holds, and their rows in one gather.

        Inputs: keys (distinct), device (where hits should land).
        Output: StoreLookup; hits (len(hit_positions), width) in the
            storage dtype, on device.
        Side effects: tells the policy which keys were hit; counts hits;
            a pinned slab gathers into a fresh pinned buffer.
        Exceptions: none.

        Example:
            >>> slab.lookup([held_key, new_key], torch.device("cuda")).miss_positions
            [1]
        """
        hit_positions: list[int] = []
        hit_rows: list[int] = []
        miss_positions: list[int] = []
        with self._lock:
            # Split the keys on the CPU index alone
            for position, key in enumerate(keys):
                row = self._row_of.get(key)
                if row is None:
                    miss_positions.append(position)
                else:
                    hit_positions.append(position)
                    hit_rows.append(row)
            if not hit_positions:
                return StoreLookup(hit_positions, None, miss_positions)

            # Gather every hit's row at once, on the slab's stream
            self._policy.record_hits([keys[position] for position in hit_positions])
            self._hits += len(hit_positions)
            with _on_default_stream(self._cuda_device(device)):
                hits = self._gather(hit_rows, device)
            _hand_to_current_stream(hits)
        result = StoreLookup(hit_positions, hits, miss_positions)
        return result

    def put_many(
        self, keys: Sequence[CardTextKey], features: torch.Tensor, durable: bool
    ) -> int:
        """Offer rows for keys; store the unheld ones the policy admits.

        Inputs: keys (distinct), features (len(keys), width) on any device,
            any float dtype (copied to the slab's placement and dtype),
            durable (ignored: a memory tier is never durable).
        Output: int, how many keys were newly admitted.
        Side effects: allocates the slab on the first call; writes admitted
            rows; drops evicted keys.
        Exceptions: ValueError if features is not 2-D, its row count is not
            len(keys), keys repeat, or its width differs from the slab's;
            RuntimeError if the policy's decision is malformed (see
            _require_decision_fits). Either way the slab's rows are
            unchanged; after a RuntimeError the policy's own state may have
            advanced, so treat it as fatal.

        Example:
            >>> slab.put_many([key_a, key_b], torch.randn(2, 768, device="cuda"), durable=False)
            2
        """
        with self._lock:
            # Check the offer and the policy's decision before changing anything
            width = None if self._rows is None else self._rows.shape[1]
            require_offer_shape(keys, features, width)
            fresh_keys, fresh_features = self._split_unheld(keys, features)
            if not fresh_keys:
                return 0
            # Allocate first: a failed allocation must not advance the policy
            if self._rows is None:
                self._rows = self._allocate(fresh_features)
            decision = self._policy.admit(
                fresh_keys, self._free_row_count(), self._pins.keys()
            )
            self._require_decision_fits(decision, fresh_keys)
            self._refused_for_pins += decision.refused_for_pins

            # Make room, then write in one copy
            with _on_default_stream(self._cuda_device(features.device)):
                for key in decision.evicted:
                    self._released_rows.append(self._row_of.pop(key))
                if decision.admitted:
                    rows = [self._claim_row(key) for key in decision.admitted]
                    selected = _select_admitted_rows(
                        fresh_keys, decision.admitted, fresh_features
                    )
                    self._write(rows, selected)
        return len(decision.admitted)

    def pin(self, keys: Sequence[CardTextKey]) -> None:
        """Add one pin to each key; a pinned key is never evicted.

        Inputs: keys (distinct; held or not). Output: none.
        Side effects: increments the keys' pin counts.
        Exceptions: ValueError if keys repeat (nothing changes).

        Example:
            >>> slab.pin([key_a])
        """
        if len(set(keys)) != len(keys):
            raise ValueError("pin was given the same key twice")
        with self._lock:
            self._pins.update(keys)

    def unpin(self, keys: Sequence[CardTextKey]) -> None:
        """Remove one pin from each key, all or nothing.

        Inputs: keys (distinct, each pinned). Output: none.
        Side effects: decrements the keys' pin counts; a key at zero is
            forgotten and evictable again.
        Exceptions: ValueError if keys repeat or a key has no pin (nothing
            changes).

        Example:
            >>> slab.unpin([key_a])
        """
        if len(set(keys)) != len(keys):
            raise ValueError("unpin was given the same key twice")
        with self._lock:
            # Check every key before changing any count
            unpinned = [key for key in keys if key not in self._pins]
            if unpinned:
                raise ValueError(f"unpin of {len(unpinned)} keys with no pin")
            for key in keys:
                self._drop_one_pin(key)

    def tier_stats(self) -> tuple[TierStats, ...]:
        """This slab's totals. Inputs: none. Output: one TierStats.
        Side effects: none. Exceptions: none."""
        with self._lock:
            pinned_rows = sum(1 for key in self._pins if key in self._row_of)
            return (
                TierStats(
                    self._name,
                    self._hits,
                    len(self._row_of),
                    self._capacity_rows,
                    pinned_rows,
                    self._refused_for_pins,
                ),
            )

    @property
    def storage_dtype(self) -> torch.dtype:
        """The dtype rows are stored in. Inputs: none. Output: torch.dtype.
        Side effects: none. Exceptions: none."""
        return self._storage_dtype

    def close(self) -> None:
        """Nothing to release: the rows are freed with the slab. Inputs:
        none. Output: none. Side effects: none. Exceptions: none."""

    def _cuda_device(self, other: torch.device) -> torch.device | None:
        """The CUDA device this call touches, if any: the slab's own (a
        VRAM slab), else other when it is CUDA (a pinned slab's copy).

        Private helper - consumers are lookup and put_many.
        Inputs: other (the asked or incoming device). Output: torch.device
            or None. Side effects: none. Exceptions: none.
        """
        if self._rows is not None and self._rows.device.type == "cuda":
            return self._rows.device
        return other if other.type == "cuda" else None

    def _drop_one_pin(self, key: CardTextKey) -> None:
        """Decrement key's pin count, forgetting it at zero.

        Private helper - single consumer is unpin (lock held, key checked).
        Inputs: key (pinned). Output: none. Side effects: as described.
        Exceptions: none.
        """
        # Delete at zero: everywhere else, "in _pins" means "has a pin"
        if self._pins[key] == 1:
            del self._pins[key]
        else:
            self._pins[key] -= 1

    def _split_unheld(
        self, keys: Sequence[CardTextKey], features: torch.Tensor
    ) -> tuple[list[CardTextKey], torch.Tensor]:
        """The keys not held yet, and their rows of features.

        Private helper - single consumer is put_many (lock held).
        Inputs: keys, features. Output: (keys, rows), same order.
        Side effects: none. Exceptions: none.
        """
        positions = [i for i, key in enumerate(keys) if key not in self._row_of]
        if len(positions) == len(keys):
            return list(keys), features
        index = torch.tensor(positions, dtype=torch.long, device=features.device)
        return [keys[i] for i in positions], features.index_select(0, index)

    def _free_row_count(self) -> int:
        """Rows not holding a key: released ones plus never-used ones.

        Private helper - single consumer is put_many.
        Inputs: none. Output: int. Side effects: none. Exceptions: none.
        """
        never_used = self._capacity_rows - self._next_unused_row
        return len(self._released_rows) + never_used

    def _claim_row(self, key: CardTextKey) -> int:
        """Take a free row for key (a released one first) and index it.

        Private helper - single consumer is put_many.
        Inputs: key (not held). Output: int, the row.
        Side effects: records key -> row; consumes a free row.
        Exceptions: none (put_many checked there is room).
        """
        if self._released_rows:
            row = self._released_rows.pop()
        else:
            row = self._next_unused_row
            self._next_unused_row += 1
        self._row_of[key] = row
        return row

    def _allocate(self, features: torch.Tensor) -> torch.Tensor:
        """An empty (capacity_rows, width) tensor in the storage dtype:
        on features' device, or pinned on the host.

        Private helper - single consumer is put_many.
        Inputs: features (N, width). Output: Tensor.
        Side effects: allocates device or pinned host memory (pinned only
            when CUDA is available). Exceptions: torch's out-of-memory
            error if it does not fit.
        """
        shape = (self._capacity_rows, features.shape[1])
        if self._placement is SlabPlacement.PINNED_HOST:
            # Pinned memory needs a CUDA (or ROCm) runtime; plain CPU without one
            return torch.empty(
                shape, dtype=self._storage_dtype, pin_memory=torch.cuda.is_available()
            )
        return torch.empty(shape, dtype=self._storage_dtype, device=features.device)

    def _gather(self, rows: list[int], device: torch.device) -> torch.Tensor:
        """The given rows, in order, as one (len(rows), width) tensor on
        device: an index_select, through a fresh pinned buffer for a
        pinned slab.

        Private helper - single consumer is lookup (on the default stream;
        lookup then records the result on the caller's stream).
        Inputs: rows (held), device. Output: Tensor in the storage dtype.
        Side effects: copies the row indices (and the rows) between
            devices; a pinned slab's copy to a CUDA device is
            non_blocking from a fresh pinned buffer.
        Exceptions: none.
        """
        assert self._rows is not None  # a held row means the slab exists
        index = torch.tensor(rows, dtype=torch.long, device=self._rows.device)
        if self._placement is SlabPlacement.FOLLOW_FIRST_PUT:
            return self._rows.index_select(0, index).to(device)
        # Pinned: gather on the CPU into a fresh pinned buffer, then copy
        staging = torch.empty(
            (len(rows), self._rows.shape[1]),
            dtype=self._storage_dtype,
            pin_memory=self._rows.is_pinned(),
        )
        torch.index_select(self._rows, 0, index, out=staging)
        return staging.to(device, non_blocking=device.type == "cuda")

    def _write(self, rows: list[int], features: torch.Tensor) -> None:
        """Write features (copied to the slab's device and dtype) into
        rows, in order.

        Private helper - single consumer is put_many (on the default stream).
        Inputs: rows, features (len(rows), width). Output: none.
        Side effects: overwrites those slab rows. A pinned slab copies
            device features to the host with a blocking copy
            (non_blocking=False), so the row is complete before any later
            CPU gather reads it. Exceptions: none.
        """
        assert self._rows is not None  # put_many allocates before writing
        index = torch.tensor(rows, dtype=torch.long, device=self._rows.device)
        # detach: the slab is storage, never part of an autograd graph. The
        # copy to a pinned slab blocks (to() defaults to non_blocking=False)
        data = features.detach().to(self._rows.device, self._storage_dtype)
        self._rows.index_copy_(0, index, data)

    def _require_decision_fits(
        self, decision: AdmissionDecision, offered: Sequence[CardTextKey]
    ) -> None:
        """Private helper - single consumer is put_many.

        Inputs: decision, offered (the keys put_many offered the policy).
        Output: none. Side effects: none.
        Exceptions: RuntimeError if decision admits keys that are not a
            subsequence of offered (repeated, reordered or never offered),
            evicts a key twice, one the slab does not hold or a pinned
            one, or admits more keys than the free rows plus its
            evictions.
        """
        if not _is_subsequence(decision.admitted, offered):
            raise RuntimeError(
                "the slot policy admitted keys that are not a subsequence of "
                "those offered"
            )
        if len(set(decision.evicted)) != len(decision.evicted):
            raise RuntimeError("the slot policy evicted a key twice")
        if any(key not in self._row_of for key in decision.evicted):
            raise RuntimeError("the slot policy evicted a key the slab does not hold")
        if any(key in self._pins for key in decision.evicted):
            raise RuntimeError("the slot policy evicted a pinned key")
        room = self._free_row_count() + len(decision.evicted)
        if len(decision.admitted) > room:
            raise RuntimeError(
                f"the slot policy admitted {len(decision.admitted)} keys "
                f"with room for {room}"
            )


@contextlib.contextmanager
def _on_default_stream(device: torch.device | None) -> Iterator[None]:
    """Run the body on device's default CUDA stream, after the caller's
    current stream, and make the caller's stream wait for it afterwards.
    Does nothing without a CUDA device.

    Private helper - consumers are TensorFeatureSlab.lookup and put_many.
    Inputs: device (CUDA, or None). Output: a context manager.
    Side effects: stream waits and a stream switch for the body.
    Exceptions: whatever the body raises (the waits still run).
    """
    if device is None:
        yield
        return
    caller = torch.cuda.current_stream(device)
    default = torch.cuda.default_stream(device)
    default.wait_stream(caller)
    try:
        with torch.cuda.stream(default):
            yield
    finally:
        caller.wait_stream(default)


def _hand_to_current_stream(tensor: torch.Tensor) -> None:
    """Record a CUDA tensor made on the default stream on the caller's
    current stream, when that is another stream, so the caching allocator
    keeps its memory until the caller's queued work is done.

    Private helper - single consumer is TensorFeatureSlab.lookup.
    Inputs: tensor. Output: none. Side effects: record_stream on a
        non-default current stream; nothing on the CPU or the default
        stream. Exceptions: none.
    """
    if tensor.device.type != "cuda":
        return
    current = torch.cuda.current_stream(tensor.device)
    if current != torch.cuda.default_stream(tensor.device):
        tensor.record_stream(current)


def _select_admitted_rows(
    keys: Sequence[CardTextKey],
    admitted: Sequence[CardTextKey],
    features: torch.Tensor,
) -> torch.Tensor:
    """The rows of features that belong to the admitted keys, in admitted
    order.

    Private helper - single consumer is TensorFeatureSlab.put_many.
    Inputs: keys (features' row order), admitted (a subsequence of keys),
        features (len(keys), width).
    Output: Tensor (len(admitted), width), on features' device.
    Side effects: none. Exceptions: none.
    """
    # Every key admitted (the common case): a subsequence of equal length
    if len(admitted) == len(keys):
        return features
    position_of = {key: position for position, key in enumerate(keys)}
    positions = [position_of[key] for key in admitted]
    index = torch.tensor(positions, dtype=torch.long, device=features.device)
    return features.index_select(0, index)


def _is_subsequence(
    picked: Sequence[CardTextKey], offered: Sequence[CardTextKey]
) -> bool:
    """Whether picked is offered with some keys left out: same order, no
    repeats, nothing new.

    Private helper - single consumer is TensorFeatureSlab._require_decision_fits.
    Inputs: picked, offered (distinct keys). Output: bool.
    Side effects: none. Exceptions: none.
    """
    remaining = iter(offered)
    # Each `in` consumes the iterator up to the match, so order is enforced
    return all(key in remaining for key in picked)
