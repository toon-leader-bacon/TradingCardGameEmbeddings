"""NeighbourDealer: a DeckSource of teacher-neighbour pseudo-decks.

Each pseudo-deck is one anchor card plus a fresh sample of its teacher
neighbours (src/teacher/), dealt as a GenericDeck, so the existing
contrastive styles train the student to place an anchor near the cards
the teacher says it is played with.

SPLITS: one teacher per split (TeacherViews in the run config). Split
S's anchors are the confident rows of S's teacher whose card tier is S
(CardTier of the same name) and that have at least
min_neighbours_per_deck neighbours. With the default views, TRAIN
anchors are TRAIN cards of the TRAIN-only teacher, and TEST anchors are
TEST cards of the TRAIN + TEST teacher: cards the student never trains
on, scored against the neighbours (TRAIN or TEST cards) that teacher
gives them. TRAIN and TEST teachers are required (the trainer scores
TEST every round); VALIDATION without a teacher deals nothing. Every
teacher must be built under the dealer's holdout and share the game and
card binder version.

PSEUDO-DECKS: the anchor and min(max_neighbours_per_deck, its neighbour
count) neighbours, drawn without replacement on every deal. With
max_neighbours_per_deck = 1 the pseudo-deck is exactly two cards, so
SingleCardStyle's 2-card draw always pairs the anchor with a neighbour;
a larger pseudo-deck also yields neighbour-neighbour pairs.

REPRODUCIBILITY: the shuffled pass (TRAIN) permutes anchors and samples
neighbours from one generator seeded at construction. An unshuffled pass
(TEST, VALIDATION) deals anchors in teacher row order and samples
neighbours from a generator re-seeded per pass from (rng_seed, split),
so every evaluation pass deals the same pseudo-decks.
"""

from dataclasses import dataclass
from typing import Iterator, Mapping
from uuid import UUID, uuid5

import numpy as np

from src.schema.card import GenericDeck
from src.schema.game_id import GameId
from src.schema.holdout import CardTier, HoldoutSpec
from src.schema.splits import Split
from src.teacher.teacher_cache import Teacher

# Names pseudo-decks: uuid5 of the anchor under this namespace
_PSEUDO_DECK_NAMESPACE = UUID("7a1f3c0e-5b2d-4e8a-9c61-2f4d8b0e6a13")

# Each split's anchors are the cards of this tier
_ANCHOR_TIERS: Mapping[Split, CardTier] = {
    Split.TRAIN: CardTier.TRAIN,
    Split.TEST: CardTier.TEST,
    Split.VALIDATION: CardTier.VALIDATION,
}


@dataclass(frozen=True)
class StudentSettings:
    """How the student's pseudo-decks are dealt (the run config's
    `student:` section).

    min_neighbours_per_deck: anchors with fewer neighbours are not dealt.
    max_neighbours_per_deck: neighbours drawn into each pseudo-deck. Only
        1 guarantees a 2-card draw (SingleCardStyle) always holds the
        anchor.

    TODO: when `student:` gains the loss choice, move this to its own
    student_settings.py; it is only the dealer's settings today.
    """

    min_neighbours_per_deck: int = 1
    max_neighbours_per_deck: int = 1

    def __post_init__(self) -> None:
        """Raise ValueError if either neighbour count is below 1."""
        if self.min_neighbours_per_deck < 1 or self.max_neighbours_per_deck < 1:
            raise ValueError("neighbours per deck must be at least 1")


class NeighbourDealer:
    """Deals one game's teacher-neighbour pseudo-decks, each split from
    its own teacher. Satisfies DeckSource. Safe to deal from one thread at
    a time, like DeckBoxDealer.
    """

    def __init__(
        self,
        teachers: Mapping[Split, Teacher],
        holdout: HoldoutSpec,
        settings: StudentSettings,
        rng_seed: int,
    ) -> None:
        """Deal pseudo-decks from each split's teacher.

        Inputs:
            teachers: one game's Teacher per split (TRAIN and TEST
                required; VALIDATION optional).
            holdout: the run's spec (each anchor's card tier).
            settings: the run's StudentSettings.
            rng_seed: seeds the TRAIN reshuffle and neighbour draws.
        Output: none (constructor).
        Side effects: none.
        Exceptions: ValueError without a TRAIN or TEST teacher, if a
            teacher was built under another holdout (deck split or card
            tier key) or the teachers disagree on game or card binder
            version, or if TRAIN or TEST has no anchor (the trainer trains
            on one and scores the other every round).

        Example:
            >>> dealer = NeighbourDealer(
            ...     {Split.TRAIN: train_teacher, Split.TEST: test_teacher},
            ...     holdout, StudentSettings(), rng_seed=0)
        """
        # Validate inputs: the teachers describe one game under one holdout
        _require_consistent_teachers(teachers, holdout)

        # Each split's anchors, from its own teacher
        anchors_by_split: dict[Split, np.ndarray] = {}
        for split in Split:
            teacher = teachers.get(split)
            anchors_by_split[split] = (
                np.empty(0, dtype=np.int64)
                if teacher is None
                else _dealt_anchors(teacher, split, holdout, settings)
            )
        for split in (Split.TRAIN, Split.TEST):
            if len(anchors_by_split[split]) == 0:
                raise ValueError(
                    f"NeighbourDealer: no {teachers[split].table.source_game.value} "
                    f"{split.value} card has {settings.min_neighbours_per_deck} "
                    "neighbours"
                )

        self._teachers = dict(teachers)
        self._settings = settings
        self._rng_seed = rng_seed
        self._shuffle_rng = np.random.default_rng(rng_seed)
        self._anchors_by_split = anchors_by_split

    @property
    def deck_split_key(self) -> str:
        """The teachers' deck_split_key: the holdout's for this game.

        Inputs: none. Output: str. Side effects: none. Exceptions: none.
        """
        return self._teachers[Split.TRAIN].table.stamp.deck_split_key

    @property
    def source_game(self) -> GameId:
        """The teachers' game.

        Inputs: none. Output: GameId. Side effects: none. Exceptions: none.
        """
        return self._teachers[Split.TRAIN].table.source_game

    @property
    def card_binder_version(self) -> str | None:
        """The CardBinder version the teachers' deck snapshot was minted
        against.

        Inputs: none. Output: str | None. Side effects: none.
        Exceptions: none.
        """
        stamp = self._teachers[Split.TRAIN].table.stamp
        return stamp.deck_snapshot.card_snapshot.card_binder_version

    def decks_for(
        self,
        split: Split,
        decks_per_sample: int,
        shuffle: bool = True,
        keep_remainder: bool = False,
    ) -> Iterator[list[GenericDeck]]:
        """Yield split's pseudo-decks, grouped into samples.

        Inputs: as DeckBoxDealer.decks_for; one pseudo-deck per anchor.
        Output: a generator of lists of GenericDeck, each of length
            decks_per_sample, except a shorter last one when
            keep_remainder is True; nothing for a split with no teacher.
        Side effects: a shuffled pass advances the shuffle generator.
        Exceptions: ValueError if decks_per_sample <= 0.

        Example:
            >>> for deck_sample in dealer.decks_for(Split.TRAIN, 64):
            ...     ...
        """
        if decks_per_sample <= 0:
            raise ValueError("decks_per_sample must be positive")

        # This pass's anchor order and neighbour generator
        anchors = self._anchors_by_split[split]
        if len(anchors) == 0:
            return
        if shuffle:
            neighbour_rng = self._shuffle_rng
            anchors = anchors[neighbour_rng.permutation(len(anchors))]
        else:
            neighbour_rng = _pass_generator(self._rng_seed, split)

        # Deal full groups, then the remainder if asked
        teacher = self._teachers[split]
        for start in range(0, len(anchors), decks_per_sample):
            group = anchors[start : start + decks_per_sample]
            if len(group) < decks_per_sample and not keep_remainder:
                return
            yield [
                self._pseudo_deck(teacher, row, neighbour_rng) for row in group.tolist()
            ]

    def deck_count(self, split: Split) -> int:
        """Number of anchors (pseudo-decks per pass) in split.

        Inputs: split. Output: int. Side effects: none. Exceptions: none.
        """
        return len(self._anchors_by_split[split])

    def _pseudo_deck(
        self, teacher: Teacher, row: int, rng: np.random.Generator
    ) -> GenericDeck:
        """teacher row's pseudo-deck: the anchor's uuid first, then
        min(max_neighbours_per_deck, its count) of its neighbours drawn
        without replacement from rng.

        Private helper - single caller is decks_for().
        Inputs: teacher (the split's), row (an anchor's teacher row), rng.
        Output: GenericDeck (nocab_uuid from _pseudo_deck_uuid, name "",
            provenance None).
        Side effects: advances rng. Exceptions: none.
        """
        card_uuids = teacher.table.card_uuids
        neighbours = teacher.neighbours.neighbours_of(row)
        drawn = rng.choice(
            neighbours,
            size=min(self._settings.max_neighbours_per_deck, len(neighbours)),
            replace=False,
        )
        anchor_uuid = card_uuids[row]
        return GenericDeck(
            nocab_uuid=_pseudo_deck_uuid(anchor_uuid),
            source_game=teacher.table.source_game,
            name="",
            card_nocab_uuids=[anchor_uuid]
            + [card_uuids[neighbour] for neighbour in drawn.tolist()],
            provenance=None,
        )


def _require_consistent_teachers(
    teachers: Mapping[Split, Teacher], holdout: HoldoutSpec
) -> None:
    """Raise unless teachers has TRAIN and TEST teachers, every teacher
    shares the TRAIN teacher's source_game and card snapshot (so binder
    version), and
    every stamp was made under holdout: deck_split_key ==
    holdout.deck_split_key(game) and card_tier_key ==
    holdout.card_tier_key() (anchors are picked by holdout.tier_of).

    Private helper - single caller is NeighbourDealer.__init__.
    Inputs: teachers, holdout. Output: None. Side effects: none.
    Exceptions: ValueError naming the first disagreement.
    """
    missing = [
        split.value for split in (Split.TRAIN, Split.TEST) if split not in teachers
    ]
    if missing:
        raise ValueError(f"NeighbourDealer needs teachers for {missing}")
    game = teachers[Split.TRAIN].table.source_game
    card_snapshot = teachers[Split.TRAIN].table.stamp.deck_snapshot.card_snapshot
    for split, teacher in teachers.items():
        stamp = teacher.table.stamp
        if teacher.table.source_game != game:
            problem = (
                f"is a {teacher.table.source_game.value} teacher, not {game.value}"
            )
        elif stamp.deck_snapshot.card_snapshot != card_snapshot:
            problem = "was built from another card snapshot"
        elif stamp.deck_split_key != holdout.deck_split_key(game):
            problem = "splits decks under another holdout"
        elif stamp.card_tier_key != holdout.card_tier_key():
            problem = "tiers cards under another holdout"
        else:
            continue
        raise ValueError(f"NeighbourDealer: the {split.value} teacher {problem}")


def _dealt_anchors(
    teacher: Teacher, split: Split, holdout: HoldoutSpec, settings: StudentSettings
) -> np.ndarray:
    """teacher's rows that anchor split: at least
    settings.min_neighbours_per_deck neighbours (so confident; filtered
    first, from the neighbour offsets), then card tier
    _ANCHOR_TIERS[split] by holdout.tier_of, ascending.

    Private helper - single caller is NeighbourDealer.__init__.
    Inputs: teacher, split, holdout, settings.
    Output: np.ndarray int64. Side effects: none. Exceptions: none.
    """
    # Enough neighbours first: cheap, and leaves only confident rows
    counts = np.diff(teacher.neighbours.offsets)
    rows = np.flatnonzero(counts >= settings.min_neighbours_per_deck)

    # Then the split's tier, one hash per surviving row
    game = teacher.table.source_game
    card_uuids = teacher.table.card_uuids
    wanted = _ANCHOR_TIERS[split]
    keep = [holdout.tier_of(card_uuids[row], game) == wanted for row in rows.tolist()]
    return rows[np.asarray(keep, dtype=bool)].astype(np.int64)


def _pass_generator(rng_seed: int, split: Split) -> np.random.Generator:
    """A fresh generator for one unshuffled pass over split, seeded from
    (rng_seed, split), so every such pass draws the same neighbours.

    Private helper - single caller is NeighbourDealer.decks_for.
    Inputs: rng_seed, split. Output: np.random.Generator.
    Side effects: none. Exceptions: none.
    """
    return np.random.default_rng([rng_seed, list(Split).index(split)])


def _pseudo_deck_uuid(anchor_uuid: UUID) -> UUID:
    """The uuid of anchor_uuid's pseudo-decks: uuid5 of the anchor under
    this module's namespace (the same for every deal; only skip logs read
    deck uuids).

    Private helper - single caller is NeighbourDealer._pseudo_deck.
    Inputs: anchor_uuid. Output: UUID. Side effects: none.
    Exceptions: none.
    """
    return uuid5(_PSEUDO_DECK_NAMESPACE, str(anchor_uuid))
