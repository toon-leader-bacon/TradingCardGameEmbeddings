"""TeacherSettings: everything that chooses how a teacher is built, as
the run config's `teacher:` section parses it, and the digest of the part
a fitted table depends on; and TeacherViews, one TeacherSettings per
student split.

Each plug-in choice is a typed value: an enum per counting rule, an
options dataclass per teacher method (TeacherMethodOptions grows as
methods are added), and a NeighbourRule Strategy (neighbour_rules.py).
"""

import hashlib
import json
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import ClassVar, Mapping

from src.schema.game_id import GameId
from src.schema.holdout import CardTier
from src.schema.splits import Split
from src.teacher.neighbour_rules import NeighbourRule

# Where teachers are cached: <directory>/<game>/
DEFAULT_TEACHER_DIRECTORY = Path("data/teachers")


class CopyCountRule(StrEnum):
    """How one deck's copies of a card weigh in the deck x card graph."""

    PRESENCE = "presence"  # 1 per distinct card in a deck
    COPIES = "copies"  # card_counts as stored
    LOG_COPIES = "log_copies"  # 1 + ln(copies)


class DeckSizeRule(StrEnum):
    """How much one deck weighs against another."""

    UNIFORM = "uniform"  # every deck weighs the same
    INVERSE_SIZE = "inverse_size"  # a deck's card pairs share one unit of weight


@dataclass(frozen=True)
class CooccurrenceOptions:
    """How the deck x card incidence matrix is weighted before X^T X."""

    copy_count_rule: CopyCountRule = CopyCountRule.PRESENCE
    deck_size_rule: DeckSizeRule = DeckSizeRule.UNIFORM


@dataclass(frozen=True)
class PpmiSvdOptions:
    """PPMI + truncated SVD (ppmi_svd.py).

    dim: teacher vector width; capped when fitted at the game's selected
        card count minus one. context_smoothing: alpha in P(j)^alpha
        (word2vec's 0.75).
    """

    KIND: ClassVar[str] = "ppmi_svd"

    dim: int
    context_smoothing: float = 0.75

    def __post_init__(self) -> None:
        """Raise ValueError for dim <= 0 or alpha outside (0, 1]."""
        if self.dim <= 0:
            raise ValueError(f"dim must be positive, got {self.dim}")
        if not 0.0 < self.context_smoothing <= 1.0:
            raise ValueError(
                f"context_smoothing must be in (0, 1], got {self.context_smoothing}"
            )

    def payload(self) -> dict[str, object]:
        """The method's kind and options, JSON-ready (TeacherSettings.fit_digest).

        Inputs: none. Output: dict. Side effects: none. Exceptions: none.

        Example:
            >>> PpmiSvdOptions(dim=64).payload()
            {'kind': 'ppmi_svd', 'dim': 64, 'context_smoothing': 0.75}
        """
        return {
            "kind": self.KIND,
            "dim": self.dim,
            "context_smoothing": self.context_smoothing,
        }


# Every teacher method's options; a new method adds its options class
# here, with KIND and payload()
TeacherMethodOptions = PpmiSvdOptions


@dataclass(frozen=True)
class TeacherSettings:
    """How to build every game's teacher in one run.

    games: whose teachers to build (none may be held out; RunConfig checks).
    method: the fit and its options. neighbours: how each card's
        neighbours are picked.
    deck_splits: which deck splits are counted (TRAIN by default).
    card_tiers: which card tiers are kept; other cards are dropped from
        every deck before counting (TRAIN by default).
    cooccurrence: the counting rules.
    min_deck_count: rows counted in fewer selected decks are kept but not
        confident (never anchors or neighbours).
    directory: the cache root (<directory>/<game>/).
    """

    games: tuple[GameId, ...]
    method: TeacherMethodOptions
    neighbours: NeighbourRule
    deck_splits: frozenset[Split] = frozenset({Split.TRAIN})
    card_tiers: frozenset[CardTier] = frozenset({CardTier.TRAIN})
    cooccurrence: CooccurrenceOptions = CooccurrenceOptions()
    min_deck_count: int = 5
    directory: Path = DEFAULT_TEACHER_DIRECTORY

    def __post_init__(self) -> None:
        """Raise ValueError for empty or repeated games, empty splits or tiers,
        or min_deck_count < 1."""
        # Validate inputs: something to build, from something
        if not self.games:
            raise ValueError("teacher games must not be empty")
        if len(set(self.games)) != len(self.games):
            raise ValueError(f"teacher games repeat: {self.games}")
        if not self.deck_splits:
            raise ValueError("teacher deck_splits must not be empty")
        if not self.card_tiers:
            raise ValueError("teacher card_tiers must not be empty")
        if self.min_deck_count < 1:
            raise ValueError(f"min_deck_count must be >= 1, got {self.min_deck_count}")

    def fit_digest(self) -> str:
        """A stable digest of every setting a fitted TeacherTable depends
        on: deck_splits, card_tiers, cooccurrence, method (kind and
        options) and min_deck_count. Not games, neighbours or directory.

        Inputs: none. Output: str (sha256 hex).
        Side effects: none. Exceptions: none.

        Example:
            >>> settings.fit_digest() == replace(settings, games=(GameId.MTG,)).fit_digest()
            True
        """
        payload = {
            "deck_splits": sorted(split.value for split in self.deck_splits),
            "card_tiers": sorted(tier.value for tier in self.card_tiers),
            "copy_count_rule": self.cooccurrence.copy_count_rule.value,
            "deck_size_rule": self.cooccurrence.deck_size_rule.value,
            "method": self.method.payload(),
            "min_deck_count": self.min_deck_count,
        }
        # TODO: one JSON -> sha256 rule is repeated here, in
        # teacher_table.stamp_tag_of and in dojos' deck_split_tag; promote
        # one helper to schema/ in its own change (it renames the
        # dojos' deck split cache files)
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class TeacherViews:
    """One TeacherSettings per student split: the teacher whose neighbours
    that split's pseudo-decks come from (the run config's `teacher:`
    section, see config_parsing/teacher_parsing.py).

    by_split: each view's settings, TRAIN and TEST required (the trainer
    scores TEST every round), VALIDATION optional. Views may differ only
    in deck_splits and card_tiers (checked). Held read-only.
    The default views: TRAIN sees TRAIN decks and cards; TEST sees TRAIN +
    TEST decks and cards, so the student's TEST anchors are cards it
    never trained on.
    """

    by_split: Mapping[Split, TeacherSettings]

    def __post_init__(self) -> None:
        """Freeze by_split. Raise ValueError without a TRAIN or TEST view,
        or if a view differs from TRAIN's in anything but deck_splits and
        card_tiers."""
        # Validate inputs: the views the trainer needs, one fit
        missing = sorted(
            split.value
            for split in (Split.TRAIN, Split.TEST)
            if split not in self.by_split
        )
        if missing:
            raise ValueError(f"teacher views need {missing}")
        _require_one_fit_but_scope(self.by_split)
        object.__setattr__(self, "by_split", MappingProxyType(dict(self.by_split)))

    @property
    def games(self) -> tuple[GameId, ...]:
        """The games every view builds a teacher for.

        Inputs: none. Output: tuple[GameId, ...]. Side effects: none.
        Exceptions: none.
        """
        return self.by_split[Split.TRAIN].games


def _require_one_fit_but_scope(by_split: Mapping[Split, TeacherSettings]) -> None:
    """Raise unless every view equals the TRAIN view once its deck_splits
    and card_tiers are set to TRAIN's.

    Private helper - single caller is TeacherViews.__post_init__.
    Inputs: by_split (with a TRAIN view). Output: None. Side effects: none.
    Exceptions: ValueError naming the first view that differs.
    """
    train = by_split[Split.TRAIN]
    for split, view in by_split.items():
        rescoped = replace(
            view, deck_splits=train.deck_splits, card_tiers=train.card_tiers
        )
        if rescoped != train:
            raise ValueError(
                f"teacher view {split.value} differs from train in more than "
                "deck_splits and card_tiers"
            )
