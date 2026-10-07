"""KeyedCardTallies - per-column tallies of one zone, kept separately per
stratum key (e.g. (pack_number, pick_number)), then summed per (card,
key) (see draft_data/README.md).

A take rate is stratified, so every per-column tally is also filed under
the row's stratum key. This class composes one float64 CardColumnTallies
(../card_column_tallies.py) per key; summing columns per card and the
count-table output are CardColumnTallies'. Per chunk, rows are sorted by
key once and each key's per-column sums are one np.add.reduceat over its
run of rows: O(rows x columns), never a per-row Python loop.
"""

from collections.abc import Callable
from uuid import UUID

import numpy as np
import numpy.typing as npt

from src.data_refinement.metrics.seventeenlands.card_column_tallies import (
    CardColumnTallies,
)

# One stratum key: a tuple of per-row values (pack_number, rank, ...)
StratumKey = tuple[object, ...]


class KeyedCardTallies:
    """tally_count float64 tallies per (stratum key, column) of one
    zone's layout, the layout fixed by the first chunk for every key.

    float64 although every tally is a whole number: it matches game_data's
    CardCountTableMetric, so every count table writes one count dtype,
    and it is exact for any count below 2**53.
    """

    def __init__(self, owner: str, tally_count: int) -> None:
        """Start with no layout and no keys.

        Inputs: owner (named in errors), tally_count (tallies per cell).
        Output: none (constructor). Side effects: none.
        Exceptions: ValueError if tally_count < 1.
        """
        if tally_count < 1:
            raise ValueError(f"{owner}: tally_count must be >= 1")
        self._owner = owner
        self._tally_count = tally_count
        self._card_uuids: tuple[UUID, ...] | None = None
        self._stratum_count: int | None = None
        self._per_key: dict[StratumKey, CardColumnTallies[np.float64]] = {}

    def add(
        self,
        card_uuids: tuple[UUID, ...],
        strata: tuple[npt.NDArray[np.generic], ...],
        increments: npt.NDArray[np.bool_],
    ) -> None:
        """Add one chunk's per-row increments, grouped by stratum key.

        Inputs:
            card_uuids: the zone's column layout (one uuid per column).
            strata: one array per stratum, each shape (rows,); () files
                every row under the empty key.
            increments: shape (tally_count, rows, columns); True adds 1.
        Output: none.
        Side effects: adds to the per-key tallies, creating a key's
            CardColumnTallies on first sight.
        Exceptions: ValueError if card_uuids differ from the first
            chunk's (for any key), or a shape disagrees.

        Example:
            >>> tallies.add(pack.card_uuids, (pick_numbers,), np.stack([present, picked]))
        """
        # Validate the layout once for every key, and the shapes
        self._check_layout(card_uuids, strata, increments)
        if increments.shape[1] == 0:
            return

        # Each key's run of rows, summed per column in one reduceat
        keys, order, starts = _key_runs(strata, increments.shape[1])
        sums = np.add.reduceat(
            increments[:, order, :], starts, axis=1, dtype=np.float64
        )

        for key_index, key in enumerate(keys):
            tallies = self._per_key.get(key)
            if tallies is None:
                tallies = CardColumnTallies(self._owner, self._tally_count, np.float64)
                self._per_key[key] = tallies
            tallies.add(card_uuids, sums[:, key_index, :])

    def count_columns(
        self,
        key_names: tuple[str, ...],
        count_names: tuple[str, ...],
        keep: Callable[[npt.NDArray[np.float64]], bool],
    ) -> tuple[dict[str, list[object]], dict[str, npt.NDArray[np.float64]]]:
        """Every kept (card, key) as count-table columns.

        Inputs:
            key_names: the output key columns: the card column then one
                name per stratum (e.g. ("nocab_uuid", "pick_number")).
            count_names: one name per tally.
            keep: whether a (card, key)'s summed tallies earn a row.
        Output: ({key column: values}, {count column: float64 array}),
            all as long as the kept rows; the card as its uuid str,
            stratum values as given (int or str). Keys in the order they
            were first added (sorted within the chunk that first saw
            them), cards per key in first-column order.
        Side effects: none.
        Exceptions: ValueError if key_names has not one name per stratum
            plus the card, or count_names not one per tally.

        Example:
            >>> tallies.count_columns(("nocab_uuid", "pick_number"), ("in_pack", "picked"), keep)
        """
        stratum_count = len(key_names) - 1
        if self._stratum_count is not None and stratum_count != self._stratum_count:
            raise ValueError(
                f"{self._owner}: {len(key_names)} key names for "
                f"{self._stratum_count} strata plus the card"
            )
        result_keys: dict[str, list[object]] = {name: [] for name in key_names}
        count_parts: dict[str, list[npt.NDArray[np.float64]]] = {
            name: [] for name in count_names
        }

        # Each key's kept cards, its stratum values repeated per card
        for key, tallies in self._per_key.items():
            if len(key) != stratum_count:
                raise ValueError(
                    f"{self._owner}: {len(key_names)} key names for "
                    f"{len(key)} strata plus the card"
                )
            card_uuids, counts = tallies.count_columns(count_names, keep)
            result_keys[key_names[0]].extend(card_uuids)
            for name, value in zip(key_names[1:], key):
                result_keys[name].extend([value] * len(card_uuids))
            for name in count_names:
                count_parts[name].append(counts[name])

        result_counts = {
            name: np.concatenate(parts) if parts else np.zeros(0, np.float64)
            for name, parts in count_parts.items()
        }
        return result_keys, result_counts

    def _check_layout(
        self,
        card_uuids: tuple[UUID, ...],
        strata: tuple[npt.NDArray[np.generic], ...],
        increments: npt.NDArray[np.bool_],
    ) -> None:
        """Fix the layout on the first chunk; reject a different one, a
        strata array whose length is not the row count, or increments
        not shaped (tally_count, rows, len(card_uuids)).

        Inputs: card_uuids, strata, increments. Output: none.
        Side effects: stores the layout on the first call.
        Exceptions: ValueError naming the owner.
        """
        if increments.ndim != 3:
            raise ValueError(
                f"{self._owner}: increments must be (tallies, rows, columns), "
                f"got shape {increments.shape}"
            )
        expected = (self._tally_count, increments.shape[1], len(card_uuids))
        if increments.shape != expected:
            raise ValueError(
                f"{self._owner}: increments shape {increments.shape}, expected {expected}"
            )
        for stratum in strata:
            if stratum.shape != (increments.shape[1],):
                raise ValueError(
                    f"{self._owner}: a stratum has shape {stratum.shape}, "
                    f"expected ({increments.shape[1]},)"
                )
        if self._stratum_count is None:
            self._stratum_count = len(strata)
        elif len(strata) != self._stratum_count:
            raise ValueError(
                f"{self._owner}: {len(strata)} strata, first chunk had "
                f"{self._stratum_count}"
            )
        if self._card_uuids is None:
            self._card_uuids = card_uuids
        elif card_uuids != self._card_uuids:
            raise ValueError(
                f"{self._owner}: chunk columns differ from the first chunk's"
            )


def _key_runs(
    strata: tuple[npt.NDArray[np.generic], ...], rows: int
) -> tuple[list[StratumKey], npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    """Rows grouped into runs by stratum key.

    Inputs: strata (each shape (rows,)), rows (> 0; the row count, used
        when strata is empty).
    Output: (distinct keys as tuples of Python values, a row order that
        puts each key's rows together (stable), the start offset of each
        key's run in that order), keys in sorted order. Each stratum is
        factorized to int codes separately (np.unique), so a mixed int /
        str key (pack, pick, rank) sorts with one np.lexsort.
    Side effects: none. Exceptions: none.
    """
    if not strata:
        return [()], np.arange(rows), np.zeros(1, np.intp)

    # Factorize each stratum, then sort rows by the code tuple
    factorized = [np.unique(stratum, return_inverse=True) for stratum in strata]
    codes = [inverse.reshape(-1) for _, inverse in factorized]
    order = np.lexsort(codes[::-1])
    sorted_codes = np.stack([code[order] for code in codes])

    # A run starts wherever any stratum's code changes
    changes = np.any(sorted_codes[:, 1:] != sorted_codes[:, :-1], axis=0)
    starts = np.concatenate([[0], np.flatnonzero(changes) + 1]).astype(np.intp)
    keys = [
        tuple(
            _python_value(values[sorted_codes[index, start]])
            for index, (values, _) in enumerate(factorized)
        )
        for start in starts
    ]
    return keys, order, starts


def _python_value(value: object) -> object:
    """A numpy scalar as its Python value (int, str); other values as is.

    Inputs: value. Output: the Python value.
    Side effects: none. Exceptions: none.
    """
    return value.item() if isinstance(value, np.generic) else value
