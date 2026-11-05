from dataclasses import replace
from uuid import UUID, uuid5

import numpy as np
import pytest

from src.data_refinement.snapshots.stamps import CardSnapshotStamp, DeckSnapshotStamp
from src.dojos.file_managers.deck_source import DeckSource
from src.dojos.file_managers.neighbour_dealer import NeighbourDealer, StudentSettings
from src.schema.game_id import GameId
from src.schema.holdout import CardTier, HoldoutSpec
from src.schema.splits import Split
from src.teacher.neighbour_rules import TopKNeighbours
from src.teacher.teacher_cache import Teacher
from src.teacher.teacher_neighbours import TeacherNeighbours
from src.teacher.teacher_table import TeacherStamp, TeacherTable

_CARD_NAMESPACE = UUID("3c9b1e40-8a7d-4f26-b5e1-0d2c6f9a4e77")

# Half the cards TRAIN tier, half TEST; decks split 8/1/1
_HOLDOUT = HoldoutSpec(
    seed=0,
    tier_ratios=(1, 1, 0),
    held_out_games=frozenset(),
    deck_split_ratios=(8, 1, 1),
)

_CARDS = [uuid5(_CARD_NAMESPACE, str(position)) for position in range(40)]


def _tier(card_uuid: UUID) -> CardTier:
    return _HOLDOUT.tier_of(card_uuid, GameId.MTG)


def _stamp(holdout: HoldoutSpec = _HOLDOUT, binder: str = "binder-v1") -> TeacherStamp:
    return TeacherStamp(
        deck_snapshot=DeckSnapshotStamp(CardSnapshotStamp(binder, 1), "box-v1", 1),
        deck_split_key=holdout.deck_split_key(GameId.MTG),
        card_tier_key=holdout.card_tier_key(),
        fit_digest="digest",
    )


def _teacher(
    cards: list[UUID],
    neighbours_per_row: int = 2,
    stamp: TeacherStamp | None = None,
    game: GameId = GameId.MTG,
) -> Teacher:
    """A teacher over cards whose row r's neighbours are the next
    neighbours_per_row rows (wrapping), except row 0, which has none."""
    stamp = stamp if stamp is not None else _stamp()
    rows = len(cards)
    lists = [
        (
            []
            if row == 0
            else [(row + step) % rows for step in range(1, neighbours_per_row + 1)]
        )
        for row in range(rows)
    ]
    offsets = np.concatenate([[0], np.cumsum([len(each) for each in lists])])
    indices = np.array([index for each in lists for index in each], dtype=np.int32)
    table = TeacherTable(
        source_game=game,
        card_uuids=tuple(cards),
        vectors=np.ones((rows, 2), dtype=np.float32),
        deck_counts=np.full(rows, 10, dtype=np.int64),
        confident=np.ones(rows, dtype=bool),
        stamp=stamp,
    )
    neighbours = TeacherNeighbours(
        teacher_stamp=stamp,
        rule=TopKNeighbours(neighbours_per_row),
        offsets=offsets.astype(np.int64),
        indices=indices,
        cosines=np.ones(len(indices), dtype=np.float32),
    )
    return Teacher(table, neighbours)


def _train_cards() -> list[UUID]:
    return [card for card in _CARDS if _tier(card) == CardTier.TRAIN]


def _teachers(**overrides: Teacher) -> dict[Split, Teacher]:
    teachers = {Split.TRAIN: _teacher(_train_cards()), Split.TEST: _teacher(_CARDS)}
    teachers.update({Split(key): value for key, value in overrides.items()})
    return teachers


def _dealer(
    settings: StudentSettings = StudentSettings(), seed: int = 0
) -> NeighbourDealer:
    return NeighbourDealer(_teachers(), _HOLDOUT, settings, seed)


def _all_decks(dealer: NeighbourDealer, split: Split, shuffle: bool = False) -> list:
    return [
        deck
        for sample in dealer.decks_for(split, 4, shuffle=shuffle, keep_remainder=True)
        for deck in sample
    ]


class TestAnchors:
    def test_train_anchors_are_train_cards_with_neighbours(self) -> None:
        anchors = [
            deck.card_nocab_uuids[0] for deck in _all_decks(_dealer(), Split.TRAIN)
        ]

        # Row 0 of the TRAIN teacher has no neighbours, so is no anchor
        assert anchors == _train_cards()[1:]

    def test_test_anchors_are_only_test_cards(self) -> None:
        anchors = [
            deck.card_nocab_uuids[0] for deck in _all_decks(_dealer(), Split.TEST)
        ]

        expected = [card for card in _CARDS[1:] if _tier(card) == CardTier.TEST]
        assert anchors == expected
        assert anchors  # the fixture has TEST cards

    def test_split_without_a_teacher_deals_nothing(self) -> None:
        dealer = _dealer()

        assert dealer.deck_count(Split.VALIDATION) == 0
        assert list(dealer.decks_for(Split.VALIDATION, 4, keep_remainder=True)) == []

    def test_deck_count_counts_anchors(self) -> None:
        dealer = _dealer()

        assert dealer.deck_count(Split.TRAIN) == len(_train_cards()) - 1
        assert dealer.deck_count(Split.TEST) == len(_all_decks(dealer, Split.TEST))

    def test_min_neighbours_drops_anchors(self) -> None:
        # TRAIN rows have 3 neighbours, TEST rows only 2: no TEST anchor
        teachers = _teachers(train=_teacher(_train_cards(), neighbours_per_row=3))

        with pytest.raises(ValueError, match="test card has 3"):
            NeighbourDealer(teachers, _HOLDOUT, StudentSettings(3, 1), 0)

    def test_min_neighbours_keeps_rows_with_enough(self) -> None:
        teachers = _teachers(
            train=_teacher(_train_cards(), neighbours_per_row=3),
            test=_teacher(_CARDS, neighbours_per_row=3),
        )

        dealer = NeighbourDealer(teachers, _HOLDOUT, StudentSettings(3, 1), 0)

        assert dealer.deck_count(Split.TRAIN) == len(_train_cards()) - 1


class TestPseudoDecks:
    def test_default_pseudo_deck_is_anchor_and_one_neighbour(self) -> None:
        teacher = _teachers()[Split.TRAIN]
        for deck in _all_decks(_dealer(), Split.TRAIN):
            anchor_row = teacher.table.card_uuids.index(deck.card_nocab_uuids[0])
            allowed = {
                teacher.table.card_uuids[row]
                for row in teacher.neighbours.neighbours_of(anchor_row).tolist()
            }
            assert len(deck.card_nocab_uuids) == 2
            assert deck.card_nocab_uuids[1] in allowed
            assert deck.source_game == GameId.MTG
            assert deck.name == ""
            assert deck.provenance is None

    def test_max_neighbours_caps_at_the_neighbour_count(self) -> None:
        dealer = _dealer(StudentSettings(1, 5))

        sizes = {len(deck.card_nocab_uuids) for deck in _all_decks(dealer, Split.TRAIN)}

        assert sizes == {3}

    def test_neighbours_drawn_without_replacement(self) -> None:
        for deck in _all_decks(_dealer(StudentSettings(1, 2)), Split.TRAIN):
            assert len(set(deck.card_nocab_uuids)) == 3

    def test_deck_uuid_is_stable_per_anchor(self) -> None:
        first = _all_decks(_dealer(), Split.TEST)
        second = _all_decks(_dealer(seed=7), Split.TEST)

        assert [deck.nocab_uuid for deck in first] == [
            deck.nocab_uuid for deck in second
        ]
        assert len({deck.nocab_uuid for deck in first}) == len(first)


class TestDealing:
    def test_unshuffled_passes_repeat(self) -> None:
        dealer = _dealer(StudentSettings(1, 1))

        first = _all_decks(dealer, Split.TEST)
        second = _all_decks(dealer, Split.TEST)

        assert first == second

    def test_shuffled_pass_deals_every_anchor_once(self) -> None:
        dealer = _dealer()
        ordered = [deck.card_nocab_uuids[0] for deck in _all_decks(dealer, Split.TRAIN)]

        shuffled = [
            deck.card_nocab_uuids[0]
            for deck in _all_decks(dealer, Split.TRAIN, shuffle=True)
        ]

        assert sorted(shuffled) == sorted(ordered)
        assert shuffled != ordered

    def test_same_seed_same_training_passes(self) -> None:
        first = _all_decks(_dealer(seed=3), Split.TRAIN, shuffle=True)
        second = _all_decks(_dealer(seed=3), Split.TRAIN, shuffle=True)

        assert first == second

    def test_groups_and_remainder(self) -> None:
        dealer = _dealer()
        count = dealer.deck_count(Split.TRAIN)

        full = list(dealer.decks_for(Split.TRAIN, 3, shuffle=False))
        kept = list(
            dealer.decks_for(Split.TRAIN, 3, shuffle=False, keep_remainder=True)
        )

        assert [len(sample) for sample in full] == [3] * (count // 3)
        assert sum(len(sample) for sample in kept) == count

    def test_rejects_non_positive_group_size(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            next(_dealer().decks_for(Split.TRAIN, 0))


class TestDeckSourceSurface:
    def test_keys_come_from_the_teacher_stamps(self) -> None:
        dealer: DeckSource = _dealer()

        assert dealer.deck_split_key == _HOLDOUT.deck_split_key(GameId.MTG)
        assert dealer.source_game == GameId.MTG
        assert dealer.card_binder_version == "binder-v1"


class TestRejectsInconsistentTeachers:
    def test_missing_test_teacher(self) -> None:
        teachers = {Split.TRAIN: _teacher(_train_cards())}

        with pytest.raises(ValueError, match="test"):
            NeighbourDealer(teachers, _HOLDOUT, StudentSettings(), 0)

    def test_teacher_under_another_holdout(self) -> None:
        other = replace(_HOLDOUT, seed=1)
        teachers = _teachers(test=_teacher(_CARDS, stamp=_stamp(other)))

        with pytest.raises(ValueError, match="another holdout"):
            NeighbourDealer(teachers, _HOLDOUT, StudentSettings(), 0)

    def test_teacher_of_another_card_snapshot(self) -> None:
        teachers = _teachers(test=_teacher(_CARDS, stamp=_stamp(binder="binder-v2")))

        with pytest.raises(ValueError, match="card snapshot"):
            NeighbourDealer(teachers, _HOLDOUT, StudentSettings(), 0)

    def test_teacher_of_another_game(self) -> None:
        teachers = _teachers(test=_teacher(_CARDS, game=GameId.FLESH_AND_BLOOD))

        with pytest.raises(ValueError, match="flesh_and_blood"):
            NeighbourDealer(teachers, _HOLDOUT, StudentSettings(), 0)


class TestStudentSettings:
    @pytest.mark.parametrize("counts", [(0, 1), (1, 0)])
    def test_rejects_counts_below_one(self, counts: tuple[int, int]) -> None:
        with pytest.raises(ValueError):
            StudentSettings(*counts)
