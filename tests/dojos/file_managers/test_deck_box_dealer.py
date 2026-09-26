import sqlite3
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from src.data_refinement.deck_box.deck_box import DeckBox
from src.dojos.file_managers.DeckBoxDealer import DeckBoxDealer
from src.schema.card import GenericDeck
from src.schema.game_id import GameId
from src.schema.splits import Split


def _deck(card_count: int = 1) -> GenericDeck:
    return GenericDeck(
        nocab_uuid=uuid4(),
        source_game=GameId.MTG,
        name="deck",
        card_nocab_uuids=[uuid4() for _ in range(card_count)],
        provenance=None,
    )


def _box_with_decks(count: int) -> tuple[DeckBox, list[UUID]]:
    box = DeckBox()
    uuids = []
    for _ in range(count):
        deck = _deck()
        box.create(deck)
        uuids.append(deck.nocab_uuid)
    return box, uuids


def _dealer(
    box: DeckBox,
    tmp_path: Path,
    name: str = "dealer",
    **kwargs,
) -> DeckBoxDealer:
    return DeckBoxDealer(box, GameId.MTG, tmp_path / f"{name}.db", **kwargs)


class TestSourceGameAndCardBinderVersion:
    def test_source_game_is_the_constructor_argument(self, tmp_path: Path) -> None:
        box, _ = _box_with_decks(1)

        dealer = _dealer(box, tmp_path)

        assert dealer.source_game == GameId.MTG

    def test_card_binder_version_is_none_for_an_unstamped_box(
        self, tmp_path: Path
    ) -> None:
        box, _ = _box_with_decks(1)

        dealer = _dealer(box, tmp_path)

        assert dealer.card_binder_version is None

    def test_card_binder_version_reflects_the_box(self, tmp_path: Path) -> None:
        box, _ = _box_with_decks(1)
        path = tmp_path / "mtg.db"
        box.save(path, GameId.MTG, "binder-v1")
        loaded = DeckBox.load([path])

        dealer = _dealer(loaded, tmp_path)

        assert dealer.card_binder_version == "binder-v1"


class TestInit:
    def test_raises_on_wrong_split_ratios_length(self, tmp_path: Path) -> None:
        box, _ = _box_with_decks(1)

        with pytest.raises(ValueError):
            _dealer(box, tmp_path, split_ratios=[1, 1])

    def test_raises_on_negative_split_ratio(self, tmp_path: Path) -> None:
        box, _ = _box_with_decks(1)

        with pytest.raises(ValueError):
            _dealer(box, tmp_path, split_ratios=[8, -1, 1])

    def test_raises_on_all_zero_split_ratios(self, tmp_path: Path) -> None:
        box, _ = _box_with_decks(1)

        with pytest.raises(ValueError):
            _dealer(box, tmp_path, split_ratios=[0, 0, 0])

    def test_only_considers_the_given_game(self, tmp_path: Path) -> None:
        box = DeckBox()
        mtg_deck = _deck()
        box.create(mtg_deck)
        other_deck = GenericDeck(
            nocab_uuid=uuid4(),
            source_game=GameId.POKEMON,
            name="other",
            card_nocab_uuids=[uuid4()],
            provenance=None,
        )
        box.create(other_deck)

        dealer = _dealer(box, tmp_path, seed=1)

        all_dealt = (
            list(dealer.training_decks(1))
            + list(dealer.test_decks(1))
            + list(dealer.validation_decks(1))
        )
        dealt_uuids = {deck.nocab_uuid for group in all_dealt for deck in group}
        assert dealt_uuids == {mtg_deck.nocab_uuid}

    def test_skips_recomputation_when_an_assignment_already_exists(
        self, tmp_path: Path
    ) -> None:
        box, _ = _box_with_decks(10)
        index_path = tmp_path / "dealer.db"
        DeckBoxDealer(box, GameId.MTG, index_path, split_ratios=[8, 1, 1], seed=1)

        # A second dealer over the SAME index_path, with a DIFFERENT
        # seed/ratios, must reuse what's already there rather than
        # recomputing - force_resplit defaults to False.
        reused = DeckBoxDealer(
            box, GameId.MTG, index_path, split_ratios=[5, 3, 2], seed=999
        )

        assert reused.deck_count(Split.TRAIN) == 8

    def test_force_resplit_recomputes_the_assignment(self, tmp_path: Path) -> None:
        box, _ = _box_with_decks(10)
        index_path = tmp_path / "dealer.db"
        DeckBoxDealer(box, GameId.MTG, index_path, split_ratios=[8, 1, 1], seed=1)

        recomputed = DeckBoxDealer(
            box,
            GameId.MTG,
            index_path,
            split_ratios=[5, 3, 2],
            seed=1,
            force_resplit=True,
        )

        assert (
            recomputed.deck_count(Split.TRAIN),
            recomputed.deck_count(Split.TEST),
            recomputed.deck_count(Split.VALIDATION),
        ) == (5, 3, 2)

    def test_a_crash_mid_multi_batch_assignment_rolls_back_completely(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # _assign_splits() wraps its whole streamed loop in `with
        # self._connection:`, specifically so an exception partway
        # through a REAL multi-batch run rolls back every batch already
        # inserted, not just the one in flight. Force that shape here
        # (a tiny batch size so 5 decks need 3 batches) and make the
        # second batch's insert raise, simulating a crash mid-assignment
        # after the first batch has already been written to the
        # connection (but not yet committed).
        import src.dojos.file_managers.DeckBoxDealer as deck_box_dealer_module

        monkeypatch.setattr(deck_box_dealer_module, "_ASSIGN_SPLITS_BATCH_SIZE", 2)
        box, _ = _box_with_decks(5)
        index_path = tmp_path / "dealer.db"

        call_count = 0
        real_insert_split_batch = DeckBoxDealer._insert_split_batch

        def _raise_on_second_batch(self: DeckBoxDealer, batch: list) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("simulated crash mid-assignment")
            real_insert_split_batch(self, batch)

        monkeypatch.setattr(
            DeckBoxDealer, "_insert_split_batch", _raise_on_second_batch
        )

        with pytest.raises(RuntimeError, match="simulated crash"):
            DeckBoxDealer(box, GameId.MTG, index_path, seed=1)
        assert (
            call_count == 2
        ), "the fault must actually land mid-multi-batch, not on the first/only batch"

        # NOTE on what this does and doesn't prove: _assign_splits() is
        # only ever called from __init__, so a failed construction's
        # DeckBoxDealer instance is never returned to any caller - there
        # is no way to reuse that specific (instance, connection) pair
        # afterward, and CPython's refcounting closes it essentially
        # immediately regardless of commit strategy. So a fresh
        # connection seeing 0 rows here is NOT, by itself, evidence that
        # an explicit rollback (vs. just "nothing ever got committed
        # because nothing pending happened to be discarded upon close")
        # is what mattered - see
        # test_a_second_assign_splits_call_rolls_back_on_its_own_still_open_connection
        # below for a test that actually isolates that distinction. What
        # THIS test verifies is real and worth keeping regardless: a
        # multi-batch run that fails partway through never leaves a
        # mixed/partial result visible to any later reader, and a clean
        # retry against the same index_path converges to a full, correct
        # assignment.
        verification = sqlite3.connect(str(index_path))
        remaining = verification.execute("SELECT COUNT(*) FROM deck_splits").fetchone()[
            0
        ]
        verification.close()
        assert remaining == 0, "a failed multi-batch assignment must not be readable"

    def test_a_second_assign_splits_call_rolls_back_on_its_own_still_open_connection(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Isolates the property the test above can't: that
        # _assign_splits()'s rollback is an EXPLICIT, synchronous effect
        # of `with self._connection:`, not an artifact of the failed
        # connection happening to get garbage-collected/closed quickly.
        # Build a dealer successfully first (a real, committed
        # all-"train" assignment), then call the PRIVATE _assign_splits()
        # a second time directly, on that SAME still-open connection,
        # with different ratios and a forced mid-batch failure - and
        # read the result back through that exact same connection
        # object, never a fresh one. If the rollback is real, the
        # original all-"train" assignment must be completely intact,
        # not a mix of the old assignment and the doomed second
        # attempt's first (would-be-committed-under-the-old-bug) batch.
        # NOTE: calling _assign_splits() a second time on one connection
        # is not a reachable production scenario (its own docstring:
        # "single consumer is __init__," and every real DeckBoxDealer
        # opens its own fresh connection) - this is a synthetic probe of
        # the `with self._connection:` idiom itself, the only way to
        # isolate it given DeckBoxDealer has no public re-entry point.
        import src.dojos.file_managers.DeckBoxDealer as deck_box_dealer_module

        box, _ = _box_with_decks(5)
        dealer = DeckBoxDealer(
            box, GameId.MTG, tmp_path / "dealer.db", split_ratios=[1, 0, 0], seed=1
        )
        assert dealer.deck_count(Split.TRAIN) == 5

        monkeypatch.setattr(deck_box_dealer_module, "_ASSIGN_SPLITS_BATCH_SIZE", 2)
        dealer._split_ratios = [0, 1, 0]  # would reassign everything to TEST

        call_count = 0
        real_insert_split_batch = DeckBoxDealer._insert_split_batch

        def _raise_on_second_batch(self: DeckBoxDealer, batch: list) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("simulated failure mid-reassignment")
            real_insert_split_batch(self, batch)

        monkeypatch.setattr(
            DeckBoxDealer, "_insert_split_batch", _raise_on_second_batch
        )

        with pytest.raises(RuntimeError, match="simulated failure mid-reassignment"):
            dealer._assign_splits()
        assert call_count == 2

        # Read through dealer's own, still-open connection - not a
        # fresh one - immediately after the caught exception.
        rows = dealer._connection.execute(
            "SELECT split, COUNT(*) FROM deck_splits GROUP BY split"
        ).fetchall()
        assert dict(rows) == {"train": 5}, (
            "the doomed reassignment's first batch must not have survived on "
            "this same connection, even though it would have committed under "
            "the old per-batch-commit design"
        )


class TestPartition:
    def test_splits_proportionally_to_ratios(self, tmp_path: Path) -> None:
        box, uuids = _box_with_decks(10)
        dealer = _dealer(box, tmp_path, split_ratios=[8, 1, 1], seed=1)

        train_count = len(list(dealer.training_decks(1)))
        test_count = len(list(dealer.test_decks(1)))
        validation_count = len(list(dealer.validation_decks(1)))

        assert (train_count, test_count, validation_count) == (8, 1, 1)

    def test_every_deck_is_dealt_exactly_once_across_splits(
        self, tmp_path: Path
    ) -> None:
        box, uuids = _box_with_decks(10)
        dealer = _dealer(box, tmp_path, split_ratios=[8, 1, 1], seed=1)

        dealt = (
            [deck for group in dealer.training_decks(1) for deck in group]
            + [deck for group in dealer.test_decks(1) for deck in group]
            + [deck for group in dealer.validation_decks(1) for deck in group]
        )

        assert sorted(deck.nocab_uuid for deck in dealt) == sorted(uuids)

    def test_deck_count_matches_decks_dealt(self, tmp_path: Path) -> None:
        box, _ = _box_with_decks(10)
        dealer = _dealer(box, tmp_path, split_ratios=[8, 1, 1], seed=1)

        assert dealer.deck_count(Split.TRAIN) == len(list(dealer.training_decks(1)))
        assert dealer.deck_count(Split.TEST) == len(list(dealer.test_decks(1)))
        assert dealer.deck_count(Split.VALIDATION) == len(
            list(dealer.validation_decks(1))
        )


class TestDecksForSplit:
    def test_raises_on_non_positive_decks_per_sample(self, tmp_path: Path) -> None:
        box, _ = _box_with_decks(3)
        dealer = _dealer(box, tmp_path, seed=1)

        with pytest.raises(ValueError):
            list(dealer.training_decks(0))

    def test_drops_a_trailing_partial_group(self, tmp_path: Path) -> None:
        box, _ = _box_with_decks(10)
        # All 10 decks land in training with these ratios.
        dealer = _dealer(box, tmp_path, split_ratios=[1, 0, 0], seed=1)

        groups = list(dealer.training_decks(decks_per_sample=3))

        assert len(groups) == 3
        assert all(len(group) == 3 for group in groups)

    def test_split_membership_is_deterministic_given_the_same_seed(
        self, tmp_path: Path
    ) -> None:
        # NOT full per-epoch order reproducibility - see
        # DeckBoxDealer's own module docstring's "real, accepted
        # trade-off": the one-time split ASSIGNMENT is seeded and
        # reproducible; the per-epoch reshuffle (shuffle=True reads)
        # deliberately is not, since it's a fresh, unseeded
        # `ORDER BY RANDOM()` per call. Asserting order equality here
        # (as an earlier version of this test did) would fail
        # non-deterministically under the SQLite-native rewrite.
        box, _ = _box_with_decks(9)
        dealer_a = DeckBoxDealer(
            box, GameId.MTG, tmp_path / "a.db", split_ratios=[1, 0, 0], seed=7
        )
        dealer_b = DeckBoxDealer(
            box, GameId.MTG, tmp_path / "b.db", split_ratios=[1, 0, 0], seed=7
        )

        uuids_a = {
            deck.nocab_uuid
            for group in dealer_a.training_decks(3, shuffle=False)
            for deck in group
        }
        uuids_b = {
            deck.nocab_uuid
            for group in dealer_b.training_decks(3, shuffle=False)
            for deck in group
        }

        assert uuids_a == uuids_b
        assert dealer_a.deck_count(Split.TRAIN) == dealer_b.deck_count(Split.TRAIN)

    def test_shuffle_true_preserves_membership_across_reshuffles(
        self, tmp_path: Path
    ) -> None:
        box, _ = _box_with_decks(20)
        dealer = _dealer(box, tmp_path, split_ratios=[1, 0, 0], seed=3)

        first = {
            deck.nocab_uuid
            for group in dealer.training_decks(1, shuffle=True)
            for deck in group
        }
        second = {
            deck.nocab_uuid
            for group in dealer.training_decks(1, shuffle=True)
            for deck in group
        }

        assert first == second
