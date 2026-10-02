from pathlib import Path

import pandas as pd
import pytest

from src.data_refinement.card_binder.card_binder import CardBinder
from src.dojos.generic.multi_card_option_selection.dojo import (
    MultiCardOptionSelectionDojo,
)
from src.dojos.generic.multi_group_binary_classification.dojo import (
    MultiGroupBinaryClassificationDojo,
)
from src.dojos.generic.multi_group_binary_classification.group_swap_mod import (
    GroupSwapMod,
)
from src.dojos.generic.multi_group_option_selection.dojo import (
    MultiGroupOptionSelectionDojo,
)
from src.dojos.generic.multi_group_regression.dojo import MultiGroupRegressionDojo
from src.dojos.isotropic import group_label_dojos as label_dojos
from src.dojos.isotropic import pick_dojos
from src.schema.holdout import HoldoutSpec
from tests.dojos.isotropic._group_fixtures import group_world, names

# Wiring tests: placeholder parquets, no real TRAIN calibration
pytestmark = pytest.mark.usefixtures("uncalibrated_generic_dojos")

_PLACEHOLDER = "00000000-0000-0000-0000-000000000000"

# (dojo class, cell, output stem, symmetric pair)
_LABEL_CASES = [
    (
        label_dojos.DeckPairWinnerDojo,
        MultiGroupBinaryClassificationDojo,
        "deck_pair_winner",
        True,
    ),
    (
        label_dojos.MidGameDeckPairWinnerDojo,
        MultiGroupBinaryClassificationDojo,
        "mid_game_deck_pair_winner",
        True,
    ),
    (
        label_dojos.EventualWinDojo,
        MultiGroupBinaryClassificationDojo,
        "mid_game_win_probability",
        False,
    ),
    (
        label_dojos.OpeningBuyOutcomeDojo,
        MultiGroupBinaryClassificationDojo,
        "opening_buy_outcome",
        False,
    ),
    (
        label_dojos.WinningDeckMembershipDojo,
        MultiGroupBinaryClassificationDojo,
        "winning_deck_membership",
        False,
    ),
    (
        label_dojos.KingdomEndingPileDojo,
        MultiGroupBinaryClassificationDojo,
        "kingdom_ending_pile_prediction",
        False,
    ),
    (
        label_dojos.WinningDeckCountDojo,
        MultiGroupRegressionDojo,
        "winning_deck_count",
        False,
    ),
    (
        label_dojos.DeckCardSetCopyCountDojo,
        MultiGroupRegressionDojo,
        "deck_card_set_copy_count",
        False,
    ),
]

_PICK_CASES = [
    (
        pick_dojos.KingdomOpeningBuyDojo,
        MultiCardOptionSelectionDojo,
        "kingdom_opening_buy_prediction",
    ),
    (
        pick_dojos.KingdomVetoDojo,
        MultiCardOptionSelectionDojo,
        "kingdom_veto_prediction",
    ),
    (pick_dojos.NextBuyDojo, MultiGroupOptionSelectionDojo, "mid_game_next_buy"),
    (
        pick_dojos.NextTrashedCardDojo,
        MultiGroupOptionSelectionDojo,
        "mid_game_next_trashed_card",
    ),
]


def _build(dojo_cls, tmp_path: Path, world=None):
    world = world or group_world()
    source = tmp_path / "isotropic_source.parquet"
    pd.DataFrame(
        {
            "x": [_PLACEHOLDER] * 10,
            # The split-group columns some of these dojos group by.
            "kingdom_uuid": [f"kingdom{i % 3}" for i in range(10)],
            "deck_set_uuid": [f"deck{i % 3}" for i in range(10)],
        }
    ).to_parquet(source, index=False)
    return dojo_cls(
        world.binder,
        HoldoutSpec.no_holdout(),
        world.box,
        card_embedding_size=4,
        path_to_training_data=source,
        rng_seed=0,
        strict_version_check=False,
    )


class TestGroupLabelDojos:
    @pytest.mark.parametrize("dojo_cls,cell,stem,symmetric", _LABEL_CASES)
    def test_cell_output_path_and_swap_mod(
        self, dojo_cls, cell, stem, symmetric, tmp_path: Path
    ) -> None:
        dojo = _build(dojo_cls, tmp_path)

        assert isinstance(dojo, cell)
        assert dojo_cls.OUTPUT_PATH == Path(f"data/metrics/isotropic/{stem}.parquet")
        if cell is MultiGroupBinaryClassificationDojo:
            mods = dojo.data_mod_pipeline.mods
            assert any(isinstance(mod, GroupSwapMod) for mod in mods) == symmetric

    def test_per_kingdom_and_per_deck_rows_split_by_group(self) -> None:
        grouped = {
            dojo_cls: dojo_cls.SPLIT_GROUP_COLUMN
            for dojo_cls, *_ in _LABEL_CASES
            if dojo_cls.SPLIT_GROUP_COLUMN is not None
        }

        assert grouped == {
            label_dojos.WinningDeckMembershipDojo: "kingdom_uuid",
            label_dojos.KingdomEndingPileDojo: "kingdom_uuid",
            label_dojos.WinningDeckCountDojo: "kingdom_uuid",
            label_dojos.DeckCardSetCopyCountDojo: "deck_set_uuid",
        }

    def test_deck_pair_winner_reads_the_metrics_columns(self, tmp_path: Path) -> None:
        world = group_world()
        dojo = _build(label_dojos.DeckPairWinnerDojo, tmp_path, world)
        chunk = pd.DataFrame(
            {
                "deck_uuid_lo": [world.add_group(["Witch"])],
                "deck_uuid_hi": [world.add_group(["Chapel"])],
                "lo_won": [False],
            }
        )

        [(groups, label)] = dojo.data_constructor.build(chunk, world.binder)

        assert [names(g) for g in groups] == [["Witch"], ["Chapel"]] and label == 0.0

    def test_membership_reads_card_then_kingdom(self, tmp_path: Path) -> None:
        world = group_world()
        dojo = _build(label_dojos.WinningDeckMembershipDojo, tmp_path, world)
        chunk = pd.DataFrame(
            {
                "kingdom_uuid": [world.add_group(["Witch", "Moat"])],
                "card_uuid": [world.uuid("Moat")],
                "in_winning_deck": [True],
            }
        )

        [(groups, label)] = dojo.data_constructor.build(chunk, world.binder)

        assert [names(g) for g in groups] == [["Moat"], ["Witch", "Moat"]]
        assert label == 1.0


class TestPickDojos:
    @pytest.mark.parametrize("dojo_cls,cell,stem", _PICK_CASES)
    def test_cell_and_output_path(self, dojo_cls, cell, stem, tmp_path: Path) -> None:
        assert isinstance(_build(dojo_cls, tmp_path), cell)
        assert dojo_cls.OUTPUT_PATH == Path(f"data/metrics/isotropic/{stem}.parquet")

    def test_next_buy_options_are_kingdom_plus_base_supply(
        self, tmp_path: Path
    ) -> None:
        world = group_world()
        dojo = _build(pick_dojos.NextBuyDojo, tmp_path, world)
        chunk = pd.DataFrame(
            {
                "partial_deck_uuid": [world.add_group(["Copper", "Estate"])],
                "kingdom_uuid": [world.add_group(["Witch", "Moat"])],
                "next_buy_card_uuids": [[world.uuid("Province")]],
            }
        )

        [([options, context], index)] = dojo.data_constructor.build(chunk, world.binder)

        assert names(options) == ["Witch", "Moat", *pick_dojos.BASE_SUPPLY_NAMES]
        assert names(context) == ["Copper", "Estate"]
        assert names(options)[index] == "Province"

    def test_next_trashed_options_are_the_decks_distinct_cards(
        self, tmp_path: Path
    ) -> None:
        world = group_world()
        dojo = _build(pick_dojos.NextTrashedCardDojo, tmp_path, world)
        deck = world.add_group(["Copper", "Copper", "Chapel", "Estate"])
        chunk = pd.DataFrame(
            {
                "partial_deck_uuid": [deck],
                "next_trashed_card_uuids": [[world.uuid("Estate")]],
            }
        )

        [([options, context], index)] = dojo.data_constructor.build(chunk, world.binder)

        assert names(options) == ["Copper", "Chapel", "Estate"] and index == 2
        assert len(context) == 4

    def test_missing_base_supply_card_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            pick_dojos.base_supply_uuids(CardBinder())
