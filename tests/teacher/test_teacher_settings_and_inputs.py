from dataclasses import replace
from pathlib import Path

import pytest

from src.data_refinement.snapshots.stamps import CardSnapshotStamp, DeckSnapshotStamp
from src.schema.game_id import GameId
from src.schema.holdout import CardTier
from src.schema.splits import Split
from src.teacher.neighbour_rules import TopKNeighbours
from src.teacher.teacher_settings import (
    CooccurrenceOptions,
    CopyCountRule,
    PpmiSvdOptions,
    TeacherViews,
)
from tests.teacher.teacher_fixtures import binder_and_cards, settings, teacher_inputs


class TestFitDigest:
    def test_ignores_games_neighbours_and_directory(self, tmp_path: Path) -> None:
        base = settings(tmp_path)
        other = replace(
            base,
            games=(GameId.MTG, GameId.FLESH_AND_BLOOD),
            neighbours=TopKNeighbours(9),
            directory=tmp_path / "elsewhere",
        )

        assert base.fit_digest() == other.fit_digest()

    @pytest.mark.parametrize(
        "change",
        [
            {"method": PpmiSvdOptions(dim=5)},
            {"card_tiers": frozenset({CardTier.TRAIN, CardTier.TEST})},
            {"deck_splits": frozenset({Split.TRAIN, Split.TEST})},
            {"cooccurrence": CooccurrenceOptions(copy_count_rule=CopyCountRule.COPIES)},
            {"min_deck_count": 9},
        ],
    )
    def test_changes_with_what_the_table_depends_on(
        self, tmp_path: Path, change: dict
    ) -> None:
        base = settings(tmp_path)

        assert base.fit_digest() != replace(base, **change).fit_digest()


class TestTeacherSettings:
    @pytest.mark.parametrize(
        "change",
        [
            {"games": ()},
            {"games": (GameId.MTG, GameId.MTG)},
            {"deck_splits": frozenset()},
            {"card_tiers": frozenset()},
            {"min_deck_count": 0},
        ],
    )
    def test_rejects_invalid_settings(self, tmp_path: Path, change: dict) -> None:
        with pytest.raises(ValueError):
            settings(tmp_path, **change)


class TestTeacherInputs:
    def test_rejects_rows_of_another_snapshot(self, tmp_path: Path) -> None:
        binder, cards = binder_and_cards(2)
        inputs = teacher_inputs(tmp_path, binder, [cards])
        other = DeckSnapshotStamp(CardSnapshotStamp("other", 1), "other", 1)

        with pytest.raises(ValueError, match="another"):
            replace(inputs, deck_rows_snapshot=other)

    def test_rejects_another_split_key(self, tmp_path: Path) -> None:
        binder, cards = binder_and_cards(2)
        inputs = teacher_inputs(tmp_path, binder, [cards])

        with pytest.raises(ValueError, match="split by"):
            replace(inputs, deck_split_key="other")

    def test_rejects_a_missing_split(self, tmp_path: Path) -> None:
        binder, cards = binder_and_cards(2)
        inputs = teacher_inputs(tmp_path, binder, [cards])
        rows = {Split.TRAIN: inputs.deck_rows_by_split[Split.TRAIN]}

        with pytest.raises(ValueError, match="no deck rows"):
            replace(inputs, deck_rows_by_split=rows)

    def test_card_mask_never_keeps_the_unknown_sentinel(self, tmp_path: Path) -> None:
        binder, cards = binder_and_cards(3)
        inputs = teacher_inputs(tmp_path, binder, [cards])

        mask = inputs.selected_card_mask(frozenset({CardTier.TRAIN}))

        assert not mask[inputs.decks.card_snapshot.unknown_index]
        assert int(mask.sum()) == 3


class TestTeacherViews:
    def _views(self, tmp_path: Path, **overrides: object) -> dict:
        train = settings(tmp_path)
        test = replace(
            train,
            deck_splits=frozenset({Split.TRAIN, Split.TEST}),
            card_tiers=frozenset({CardTier.TRAIN, CardTier.TEST}),
        )
        views = {Split.TRAIN: train, Split.TEST: test}
        views.update({Split(key): value for key, value in overrides.items()})
        return views

    def test_games_come_from_the_views(self, tmp_path: Path) -> None:
        views = TeacherViews(self._views(tmp_path))

        assert views.games == (GameId.MTG,)

    def test_views_are_read_only(self, tmp_path: Path) -> None:
        views = TeacherViews(self._views(tmp_path))

        with pytest.raises(TypeError):
            views.by_split[Split.VALIDATION] = views.by_split[Split.TRAIN]  # type: ignore[index]

    @pytest.mark.parametrize("missing", [Split.TRAIN, Split.TEST])
    def test_train_and_test_are_required(self, tmp_path: Path, missing: Split) -> None:
        by_split = self._views(tmp_path)
        del by_split[missing]

        with pytest.raises(ValueError, match=missing.value):
            TeacherViews(by_split)

    def test_views_may_differ_only_in_scope(self, tmp_path: Path) -> None:
        other_fit = replace(settings(tmp_path), method=PpmiSvdOptions(dim=9))

        with pytest.raises(ValueError, match="test differs"):
            TeacherViews(self._views(tmp_path, test=other_fit))
