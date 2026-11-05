"""Helpers building real deck snapshots and TeacherInputs from small,
hand-written decks, for the teacher tests."""

from dataclasses import replace
from pathlib import Path
from uuid import UUID, uuid5

import numpy as np

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.snapshots.stamps import CardSnapshotStamp, DeckSnapshotStamp
from src.dojos.file_managers.deck_split_index import DeckSplitIndex
from src.schema.card import GenericDeck
from src.schema.game_id import GameId
from src.schema.holdout import HoldoutSpec
from src.schema.splits import Split
from src.teacher.neighbour_rules import TopKNeighbours
from src.teacher.teacher_inputs import TeacherInputs
from src.teacher.teacher_settings import PpmiSvdOptions, TeacherSettings
from src.teacher.teacher_table import TeacherStamp, TeacherTable
from tests.snapshot_fixtures import built_snapshots, card, stamped_deck_box

# Seeds deck uuids, so deck splits are the same on every run
_DECK_NAMESPACE = UUID("6f1c2a52-3d0e-4b8e-9a51-7e0f3f6c2b10")
_CARD_NAMESPACE = UUID("0b7d6a9e-1c4f-4e2a-8d3b-5a6f7c8e9d01")


def all_train_holdout(
    held_out_games: frozenset[GameId] = frozenset(),
) -> HoldoutSpec:
    """Every card TRAIN tier and every deck TRAIN split."""
    return HoldoutSpec(
        seed=0,
        tier_ratios=(1, 0, 0),
        held_out_games=held_out_games,
        deck_split_ratios=(1, 0, 0),
    )


def binder_and_cards(count: int) -> tuple[CardBinder, list[UUID]]:
    """A binder of count MTG cards with fixed uuids (card tiers hash the
    uuid, so tests stay reproducible), and those uuids in order."""
    binder = CardBinder()
    cards: list[UUID] = []
    for position in range(count):
        fresh = card(GameId.MTG, f"card {position}")
        fixed = replace(fresh, nocab_uuid=uuid5(_CARD_NAMESPACE, str(position)))
        binder.create(fixed)
        cards.append(fixed.nocab_uuid)
    return binder, cards


def teacher_inputs(
    tmp_path: Path,
    binder: CardBinder,
    deck_cards: list[list[UUID]],
    holdout: HoldoutSpec | None = None,
) -> TeacherInputs:
    """TeacherInputs over one MTG deck per deck_cards entry (duplicates in
    an entry are copies), through real snapshots and a split index."""
    holdout = holdout if holdout is not None else all_train_holdout()
    decks = [
        GenericDeck(uuid5(_DECK_NAMESPACE, str(position)), GameId.MTG, "deck", cards)
        for position, cards in enumerate(deck_cards)
    ]
    box = stamped_deck_box(decks, binder, tmp_path)
    snapshot = built_snapshots(binder, box, tmp_path)
    split_index = DeckSplitIndex.load_or_build(snapshot, holdout, tmp_path / "splits")
    return TeacherInputs(
        decks=snapshot,
        deck_rows_by_split={split: split_index.rows_of(split) for split in Split},
        deck_rows_snapshot=split_index.snapshot_stamp,
        deck_split_key=split_index.deck_split_key,
        holdout=holdout,
    )


def settings(directory: Path, **overrides: object) -> TeacherSettings:
    """Small MTG teacher settings caching under directory."""
    values: dict[str, object] = {
        "games": (GameId.MTG,),
        "method": PpmiSvdOptions(dim=4),
        "neighbours": TopKNeighbours(k=2),
        "min_deck_count": 1,
        "directory": directory,
    }
    values.update(overrides)
    return TeacherSettings(**values)  # type: ignore[arg-type]


def stamp(fit_digest: str = "digest") -> TeacherStamp:
    """A TeacherStamp naming made-up versions."""
    return TeacherStamp(
        deck_snapshot=DeckSnapshotStamp(CardSnapshotStamp("binder-v1", 1), "box-v1", 1),
        deck_split_key="split-key",
        card_tier_key="tier-key",
        fit_digest=fit_digest,
    )


def table_of(
    vectors: list[list[float]], confident: list[bool] | None = None
) -> TeacherTable:
    """A TeacherTable over made-up cards with the given vectors."""
    rows = len(vectors)
    return TeacherTable(
        source_game=GameId.MTG,
        card_uuids=tuple(uuid5(_DECK_NAMESPACE, f"card {row}") for row in range(rows)),
        vectors=np.array(vectors, dtype=np.float32),
        deck_counts=np.arange(rows, dtype=np.int64) + 1,
        confident=np.array(confident if confident is not None else [True] * rows),
        stamp=stamp(),
    )
