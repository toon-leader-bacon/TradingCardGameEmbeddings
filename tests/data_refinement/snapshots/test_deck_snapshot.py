import logging
from collections import Counter
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np
import pytest

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.deck_box.deck_lookup import DeckCardCounts
from src.data_refinement.snapshots.build import (
    build_card_snapshot,
    build_deck_snapshot,
)
from src.data_refinement.snapshots.card_snapshot import CardSnapshot
from src.data_refinement.snapshots.deck_snapshot import DeckSnapshot
from src.data_refinement.snapshots.stamps import (
    StaleDeckBoxError,
    StaleSnapshotError,
)
from src.schema.card import GenericDeck
from src.schema.game_id import GameId
from tests.snapshot_fixtures import (
    binder_with_cards,
    built_snapshots,
    card,
    stamped_deck_box,
)


def _deck(card_uuids: list[UUID]) -> GenericDeck:
    return GenericDeck(
        nocab_uuid=uuid4(),
        source_game=GameId.MTG,
        name="deck",
        card_nocab_uuids=card_uuids,
        provenance=None,
    )


def _multiset_of(snapshot: DeckSnapshot, row: int) -> Counter[UUID]:
    start, end = snapshot.deck_offsets[row], snapshot.deck_offsets[row + 1]
    uuids = snapshot.card_snapshot.card_uuids
    return Counter(
        {
            uuids[int(index)]: int(count)
            for index, count in zip(
                snapshot.card_indices[start:end], snapshot.card_counts[start:end]
            )
        }
    )


def _row_of(snapshot: DeckSnapshot, deck_uuid: UUID) -> int:
    rows = [row.tobytes() for row in snapshot.deck_uuid_bytes]
    return rows.index(deck_uuid.bytes)


class TestContents:
    def test_every_deck_keeps_its_card_multiset(self, tmp_path: Path) -> None:
        binder = binder_with_cards(8)
        cards = list(binder.all_uuids(GameId.MTG))
        decks = [_deck([cards[0], cards[0], cards[1]]), _deck(cards[2:8])]
        box = stamped_deck_box(decks, binder, tmp_path)

        snapshot = built_snapshots(binder, box, tmp_path)

        assert snapshot.deck_count == 2
        for deck in decks:
            assert _multiset_of(snapshot, _row_of(snapshot, deck.nocab_uuid)) == (
                Counter(deck.card_nocab_uuids)
            )

    def test_rows_are_in_deck_uuid_byte_order(self, tmp_path: Path) -> None:
        binder = binder_with_cards(2)
        decks = [_deck(list(binder.all_uuids(GameId.MTG))) for _ in range(15)]
        box = stamped_deck_box(decks, binder, tmp_path)

        snapshot = built_snapshots(binder, box, tmp_path)

        rows = [row.tobytes() for row in snapshot.deck_uuid_bytes]
        assert rows == sorted(deck.nocab_uuid.bytes for deck in decks)

    def test_entries_ascend_by_card_index_within_a_deck(self, tmp_path: Path) -> None:
        binder = binder_with_cards(12)
        box = stamped_deck_box(
            [_deck(list(binder.all_uuids(GameId.MTG)))], binder, tmp_path
        )

        snapshot = built_snapshots(binder, box, tmp_path)

        assert np.all(np.diff(snapshot.card_indices.astype(np.int64)) > 0)

    def test_arrays_have_the_documented_types(self, tmp_path: Path) -> None:
        binder = binder_with_cards(2)
        box = stamped_deck_box(
            [_deck(list(binder.all_uuids(GameId.MTG)))], binder, tmp_path
        )

        snapshot = built_snapshots(binder, box, tmp_path)

        assert snapshot.deck_uuid_bytes.dtype == np.uint8
        assert snapshot.deck_offsets.dtype == np.int32
        assert snapshot.card_indices.dtype == np.uint16
        assert snapshot.card_counts.dtype == np.uint8
        assert not snapshot.card_indices.flags.writeable

    def test_a_deck_with_no_cards_has_no_entries(self, tmp_path: Path) -> None:
        binder = binder_with_cards(1)
        box = stamped_deck_box([_deck([])], binder, tmp_path)

        snapshot = built_snapshots(binder, box, tmp_path)

        assert snapshot.deck_count == 1
        assert list(snapshot.deck_offsets) == [0, 0]

    def test_an_empty_box_gives_an_empty_snapshot(self, tmp_path: Path) -> None:
        binder = binder_with_cards(1)
        box = stamped_deck_box([], binder, tmp_path)

        snapshot = built_snapshots(binder, box, tmp_path)

        assert snapshot.deck_count == 0
        assert list(snapshot.deck_offsets) == [0]
        assert snapshot.deck_uuid_bytes.shape == (0, 16)


class TestIrregularEntries:
    def test_counts_above_255_clip_with_one_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        binder = binder_with_cards(2)
        big, small = list(binder.all_uuids(GameId.MTG))
        decks = [_deck([big] * 300 + [small] * 255), _deck([big] * 1000)]
        box = stamped_deck_box(decks, binder, tmp_path)

        with caplog.at_level(logging.WARNING):
            snapshot = built_snapshots(binder, box, tmp_path)

        assert _multiset_of(snapshot, _row_of(snapshot, decks[0].nocab_uuid)) == (
            Counter({big: 255, small: 255})
        )
        warnings = [r.getMessage() for r in caplog.records if "clipped" in r.message]
        assert warnings == [
            "mtg: clipped 2 deck entries to a count of 255 (largest count 1000); "
            "the deck box keeps the true counts"
        ]

    def test_unknown_cards_merge_into_the_sentinel_with_a_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        binder = binder_with_cards(1)
        (known,) = binder.all_uuids(GameId.MTG)
        deck = _deck([known, uuid4(), uuid4(), uuid4()])
        box = stamped_deck_box([deck], binder, tmp_path)

        with caplog.at_level(logging.WARNING):
            snapshot = built_snapshots(binder, box, tmp_path)

        assert _multiset_of(snapshot, 0) == Counter(
            {known: 1, CardBinder.unknown_card_uuid(GameId.MTG): 3}
        )
        assert any("3 deck cards are not in" in r.message for r in caplog.records)

    def test_a_clean_build_logs_no_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        binder = binder_with_cards(2)
        box = stamped_deck_box(
            [_deck(list(binder.all_uuids(GameId.MTG)))], binder, tmp_path
        )

        with caplog.at_level(logging.WARNING):
            built_snapshots(binder, box, tmp_path)

        assert caplog.records == []


class _RepeatingDeckBox:
    """A deck box stand-in that yields one deck twice."""

    def __init__(self, box: DeckBox) -> None:
        self._box = box

    def __getattr__(self, name: str) -> object:
        return getattr(self._box, name)

    def deck_card_counts(self, source_game: GameId) -> list[DeckCardCounts]:
        decks = list(self._box.deck_card_counts(source_game))
        return decks + decks[:1]


class TestDeckOrder:
    def test_decks_read_in_any_order_are_stored_by_uuid(self, tmp_path: Path) -> None:
        binder = binder_with_cards(5)
        card_uuids = list(binder.all_uuids(GameId.MTG))
        decks = [_deck(card_uuids[: 1 + position % 5]) for position in range(30)]
        box = stamped_deck_box(decks, binder, tmp_path)

        snapshot = built_snapshots(binder, box, tmp_path)

        for deck in decks:
            assert _multiset_of(snapshot, _row_of(snapshot, deck.nocab_uuid)) == (
                Counter(deck.card_nocab_uuids)
            )

    def test_a_deck_read_twice_is_refused(self, tmp_path: Path) -> None:
        binder = binder_with_cards(1)
        box = stamped_deck_box([_deck([]), _deck([])], binder, tmp_path)
        build_card_snapshot(binder, GameId.MTG, tmp_path / "cards")
        cards = CardSnapshot.open(tmp_path / "cards", binder, GameId.MTG)

        with pytest.raises(ValueError, match="read twice"):
            build_deck_snapshot(
                _RepeatingDeckBox(box), cards, tmp_path / "decks"  # type: ignore[arg-type]
            )


class TestMintedAgainstAnotherBinder:
    def test_an_unstamped_box_is_refused(self, tmp_path: Path) -> None:
        binder = binder_with_cards(1)
        build_card_snapshot(binder, GameId.MTG, tmp_path / "cards")
        cards = CardSnapshot.open(tmp_path / "cards", binder, GameId.MTG)
        box = DeckBox()
        box.create(_deck([]))

        with pytest.raises(StaleDeckBoxError, match="None"):
            build_deck_snapshot(box, cards, tmp_path / "decks")
        assert not (tmp_path / "decks").exists()

    def test_a_box_minted_against_another_binder_is_refused(
        self, tmp_path: Path
    ) -> None:
        binder = binder_with_cards(1)
        box = stamped_deck_box([_deck([])], binder, tmp_path)
        binder.create(card())
        build_card_snapshot(binder, GameId.MTG, tmp_path / "cards")
        cards = CardSnapshot.open(tmp_path / "cards", binder, GameId.MTG)

        with pytest.raises(StaleDeckBoxError, match="run_deck_box_ingestion"):
            build_deck_snapshot(box, cards, tmp_path / "decks")


class TestStale:
    def test_a_missing_file_is_stale(self, tmp_path: Path) -> None:
        binder = binder_with_cards(1)
        build_card_snapshot(binder, GameId.MTG, tmp_path / "cards")
        cards = CardSnapshot.open(tmp_path / "cards", binder, GameId.MTG)
        box = stamped_deck_box([], binder, tmp_path)

        with pytest.raises(StaleSnapshotError, match="build_training_snapshots"):
            DeckSnapshot.open(tmp_path / "decks", cards, box)

    def test_a_grown_deck_box_is_stale(self, tmp_path: Path) -> None:
        binder = binder_with_cards(1)
        box = stamped_deck_box([_deck([])], binder, tmp_path)
        snapshot = built_snapshots(binder, box, tmp_path)
        grown = stamped_deck_box([_deck([]), _deck([])], binder, tmp_path / "grown")

        with pytest.raises(StaleSnapshotError, match="another deck box version"):
            DeckSnapshot.open(tmp_path / "decks", snapshot.card_snapshot, grown)

    def test_a_rebuilt_card_snapshot_is_stale(self, tmp_path: Path) -> None:
        binder = binder_with_cards(1)
        box = stamped_deck_box([_deck([])], binder, tmp_path)
        built_snapshots(binder, box, tmp_path)
        other_binder = binder_with_cards(2)
        build_card_snapshot(other_binder, GameId.MTG, tmp_path / "other_cards")
        other_cards = CardSnapshot.open(
            tmp_path / "other_cards", other_binder, GameId.MTG
        )

        with pytest.raises(StaleSnapshotError, match="another card snapshot"):
            DeckSnapshot.open(tmp_path / "decks", other_cards, box)

    def test_default_directory_is_per_game(self) -> None:
        assert DeckSnapshot.default_directory(GameId.MTG) == Path(
            "data/derived/deck_snapshots/mtg"
        )
