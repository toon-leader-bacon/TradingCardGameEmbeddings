import inspect
import random
from pathlib import Path
from typing import Any

import pytest

from src.data_refinement.card_binder.card_binder import CardBinder
from src.dojos.generic.generic_dojo import GenericDojo
from src.dojos.generic.multi_card_binary_classification.dojo import (
    MultiCardBinaryClassificationDojo,
)
from src.dojos.generic.multi_card_fixed_classification.dojo import (
    MultiCardFixedClassificationDojo,
)
from src.dojos.generic.multi_card_option_selection.dojo import (
    MultiCardOptionSelectionDojo,
)
from src.dojos.generic.multi_card_regression.dojo import MultiCardRegressionDojo
from src.dojos.generic.multi_group_binary_classification.dojo import (
    MultiGroupBinaryClassificationDojo,
)
from src.dojos.generic.multi_group_option_selection.dojo import (
    MultiGroupOptionSelectionDojo,
)
from src.dojos.generic.multi_group_regression.dojo import MultiGroupRegressionDojo
from src.dojos.loss.regression_objective import RegressionObjective
from src.dojos.mods.common_mods import MaskTargetKeyMod
from src.dojos.mods.mod_pipeline import ModPipeline
from src.dojos.mods.per_game_mod import PerGameMod
from src.dojos.cross_game.rarity_tier_dojo import RarityTierDojo
from src.data_refinement.metrics.cross_game.rarity.rarity_tier_metric import (
    RarityTierMetric,
)
from src.dojos.mods.mod_specs import (
    CardDropoutSpec,
    DuplicateCollapseSpec,
    ShuffleKeysSpec,
)
from src.data_refinement.metrics.dominiontabs.cost_regression_metric import (
    CostRegressionMetric,
)
from src.data_refinement.metrics.dominiontabs.set_mask_metric import (
    SetMaskMetric as DominionSetMaskMetric,
)
from src.data_refinement.metrics.isotropic.games.kingdom_ending_type_metric import (
    KingdomEndingTypeMetric,
)
from src.data_refinement.metrics.isotropic.summary.deck_card_mask_metric import (
    WinningDeckMaskedCardMetric,
)
from src.data_refinement.metrics.isotropic.summary.full_deck_win_prediction_metric import (  # noqa: E501
    FullDeckWinPredictionMetric,
)
from src.data_refinement.metrics.play_gwent.leader_masked_from_deck_metric import (
    LeaderMaskedFromDeckMetric,
)
from src.data_refinement.metrics.seventeenlands.draft_data.pack_to_pick_choice_set_metric import (  # noqa: E501
    PackToPickChoiceSetMetric,
)
from src.data_refinement.metrics.seventeenlands.draft_data.pick_number_decay_curve_metric import (  # noqa: E501
    PickNumberDecayCurveMetric,
)
from src.data_refinement.metrics.seventeenlands.draft_data.pool_conditioned_pick_metric import (  # noqa: E501
    PoolConditionedPickMetric,
)
from src.data_refinement.metrics.seventeenlands.game_data.game_deck_label_metrics import (  # noqa: E501
    DeckRankTierPredictionMetric,
    DeckWinPredictionMetric,
)
from src.data_refinement.metrics.seventeenlands.replay_data.attacker_blocker_combat_outcome_metric import (  # noqa: E501
    AttackerBlockerCombatOutcomeMetric,
)
from src.data_refinement.metrics.sts2_runs import card_average_metrics as sts2_cards
from src.data_refinement.metrics.sts2_runs import deck_label_metrics as sts2_decks
from src.data_refinement.metrics.sts_gg.card_character_prediction_metric import (
    CardCharacterPredictionMetric,
)
from src.dojos.sts_gg.card_character_prediction_dojo import (
    CardCharacterPredictionDojo,
)
from src.data_refinement.metrics.sts_gg.deck_label_metrics import (
    CharacterPredictionMetric,
    KilledByMetric,
    WinMetric,
)
from src.dojos import isotropic
from src.dojos.dominiontabs.cost_regression_dojo import CostRegressionDojo
from src.dojos.dominiontabs.masked_field_dojos import SetMaskDojo
from src.dojos.play_gwent.deck_card_mask_dojos import LeaderMaskedFromDeckDojo
from src.dojos.seventeenlands.draft_data.pack_to_pick_choice_set_dojo import (
    PackToPickChoiceSetDojo,
)
from src.dojos.seventeenlands.draft_data.pick_number_decay_curve_dojo import (
    PickNumberDecayCurveDojo,
)
from src.dojos.seventeenlands.draft_data.pool_conditioned_pick_dojo import (
    PoolConditionedPickDojo,
)
from src.dojos.seventeenlands.game_data.game_deck_label_dojos import (
    DeckRankTierPredictionDojo,
    DeckWinPredictionDojo,
)
from src.dojos.seventeenlands.replay_data.attacker_blocker_combat_outcome_dojo import (  # noqa: E501
    AttackerBlockerCombatOutcomeDojo,
)
from src.dojos.sts_gg import deck_label_dojos
from src.schema.game_id import GameId
from src.schema.holdout import HoldoutSpec
from src.training.dojo_catalog import (
    DECK_MOD_GROUPS,
    DOJO_CATALOG,
    CardDojoRecipe,
    CardShelf,
    ContrastiveDojoRecipe,
    DeckDojoRecipe,
    DojoBuildContext,
    MultiGameCardDojoRecipe,
    _REGRESSION_CELLS,
    _augmentation_pipeline,
    _constructor_for,
    _staple_subsampling,
    _with_augmentations,
    build_dojos,
)


class _RecordingShelf(CardShelf):
    """Hands out sentinel objects instead of loading files."""

    def __init__(self) -> None:
        super().__init__()
        self.requested_games: list[GameId] = []
        self.requested_boxes: list[Path] = []

    def card_binder(self, game: GameId) -> Any:
        self.requested_games.append(game)
        return f"binder:{game.value}"

    def card_lookup(self, games: Any) -> Any:
        self.requested_games.extend(games)
        return "lookup:" + ",".join(game.value for game in games)

    def deck_box(self, path: Path) -> Any:
        self.requested_boxes.append(path)
        return f"box:{path.name}"


def _stub_dojo(name: str, mods: list | None = None) -> GenericDojo:
    """A GenericDojo shell (no data) with a name and a task mod pipeline."""
    dojo = GenericDojo.__new__(GenericDojo)
    dojo.name = name
    dojo.data_mod_pipeline = ModPipeline(mods or [])
    return dojo


class _RecordingDojoClass:
    """Stands in for a dojo class; records the arguments it was built with
    and returns a stub GenericDojo named by its name argument."""

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append((args, kwargs))
        return _stub_dojo(kwargs["name"])


# Dojos that name their metric in their constructor instead of a METRIC ClassVar
_INLINE_METRICS: dict[Any, Any] = {
    deck_label_dojos.WinDojo: WinMetric,
    deck_label_dojos.KilledByDojo: KilledByMetric,
    deck_label_dojos.CharacterDojo: CharacterPredictionMetric,
    CardCharacterPredictionDojo: CardCharacterPredictionMetric,
    RarityTierDojo: RarityTierMetric,
    CostRegressionDojo: CostRegressionMetric,
    SetMaskDojo: DominionSetMaskMetric,
    LeaderMaskedFromDeckDojo: LeaderMaskedFromDeckMetric,
    isotropic.FullDeckWinPredictionDojo: FullDeckWinPredictionMetric,
    isotropic.KingdomEndingTypeDojo: KingdomEndingTypeMetric,
    isotropic.WinningDeckMaskedCardDojo: WinningDeckMaskedCardMetric,
    PackToPickChoiceSetDojo: PackToPickChoiceSetMetric,
    PickNumberDecayCurveDojo: PickNumberDecayCurveMetric,
    PoolConditionedPickDojo: PoolConditionedPickMetric,
    DeckWinPredictionDojo: DeckWinPredictionMetric,
    DeckRankTierPredictionDojo: DeckRankTierPredictionMetric,
    AttackerBlockerCombatOutcomeDojo: AttackerBlockerCombatOutcomeMetric,
}


# Every sts2_runs metric class, by the output path a recipe can override with
_METRICS_BY_OUTPUT: dict[Path, Any] = {
    cls.DEFAULT_OUTPUT_PATH: cls
    for module in (sts2_cards, sts2_decks)
    for cls in vars(module).values()
    if isinstance(cls, type)
    and isinstance(getattr(cls, "DEFAULT_OUTPUT_PATH", None), Path)
}


def _metric_output_path_of(dojo_class: Any) -> Path:
    if hasattr(dojo_class, "OUTPUT_PATH"):  # isotropic dojos name the path
        return dojo_class.OUTPUT_PATH
    metric = _INLINE_METRICS.get(dojo_class) or dojo_class.METRIC
    return metric.DEFAULT_OUTPUT_PATH


def _sliced_metric_of(dojo_class: Any) -> Any:
    """The 17lands metric a dojo class trains on (one with a FAMILY), or
    None for any other dojo class."""
    metric = _INLINE_METRICS.get(dojo_class) or getattr(dojo_class, "METRIC", None)
    return metric if hasattr(metric, "FAMILY") else None


def _context(shelf: CardShelf) -> DojoBuildContext:
    return DojoBuildContext(
        shelf=shelf,
        holdout=HoldoutSpec.no_holdout(),
        card_embedding_size=32,
        rng_seed=7,
    )


class TestRecipes:
    def test_card_recipe_passes_the_binder_name_and_seed(self) -> None:
        shelf = _RecordingShelf()
        dojo_class = _RecordingDojoClass()
        context = _context(shelf)

        CardDojoRecipe(GameId.GWENT, dojo_class).build("gwent_one.x", context)

        assert dojo_class.calls == [
            (
                ("binder:gwent", context.holdout, 32),
                {"path_to_training_data": None, "name": "gwent_one.x", "rng_seed": 7},
            )
        ]

    def test_deck_recipe_also_passes_its_deck_box(self) -> None:
        shelf = _RecordingShelf()
        dojo_class = _RecordingDojoClass()
        context = _context(shelf)
        recipe = DeckDojoRecipe(GameId.SLAY_THE_SPIRE_2, dojo_class, Path("x/box.db"))

        recipe.build("sts_gg.win", context)

        args, kwargs = dojo_class.calls[0]
        assert args == ("binder:slay_the_spire_2", context.holdout, "box:box.db", 32)
        assert kwargs == {
            "path_to_training_data": None,
            "name": "sts_gg.win",
            "rng_seed": 7,
        }
        assert shelf.requested_boxes == [Path("x/box.db")]

    def test_multi_game_recipe_builds_one_lookup_over_its_games(self) -> None:
        shelf = _RecordingShelf()
        dojo_class = _RecordingDojoClass()
        context = _context(shelf)
        recipe = MultiGameCardDojoRecipe(
            (GameId.GWENT, GameId.MTG), dojo_class  # type: ignore[arg-type]
        )

        dojo = recipe.build("cross_game.x", context)

        args, kwargs = dojo_class.calls[0]
        assert args == ("lookup:gwent,mtg", context.holdout, 32)
        assert kwargs == {"name": "cross_game.x", "rng_seed": 7}

        # Each game's augmentations ride in one PerGameMod after the task mods
        (mod,) = dojo.data_mod_pipeline.mods
        assert isinstance(mod, PerGameMod)
        assert mod.train_only
        assert set(mod._pipelines) == {GameId.GWENT, GameId.MTG}

    def test_multi_game_recipe_override_replaces_every_games_augmentations(
        self,
    ) -> None:
        context = DojoBuildContext(
            shelf=_RecordingShelf(),
            holdout=HoldoutSpec.no_holdout(),
            card_embedding_size=32,
            rng_seed=7,
            mod_overrides={"cross_game.x": (ShuffleKeysSpec(),)},
        )
        recipe = MultiGameCardDojoRecipe(
            (GameId.GWENT, GameId.MTG), _RecordingDojoClass()  # type: ignore[arg-type]
        )

        dojo = recipe.build("cross_game.x", context)

        (mod,) = dojo.data_mod_pipeline.mods
        assert isinstance(mod, PerGameMod)
        for pipeline in mod._pipelines.values():
            assert [type(inner).__name__ for inner in pipeline.mods] == [
                "ShuffleKeysMod"
            ]

    def test_metric_output_replaces_the_dojos_own_metric_file(self) -> None:
        dojo_class = _RecordingDojoClass()
        recipe = CardDojoRecipe(GameId.GWENT, dojo_class, Path("m/other.parquet"))

        recipe.build("m.other", _context(_RecordingShelf()))

        _, kwargs = dojo_class.calls[0]
        assert kwargs["path_to_training_data"] == Path("m/other.parquet")


class TestCatalog:
    @pytest.mark.parametrize(
        "key",
        sorted(
            k
            for k, r in DOJO_CATALOG.items()
            if not isinstance(r, ContrastiveDojoRecipe)
        ),
    )
    def test_each_key_names_its_metric_file(self, key: str) -> None:
        # "<source>.<stem>" must match the metric output the dojo reads, so
        # a key cannot silently point at a different metric
        recipe = DOJO_CATALOG[key]
        assert isinstance(
            recipe, (CardDojoRecipe, DeckDojoRecipe, MultiGameCardDojoRecipe)
        )
        sliced_metric = _sliced_metric_of(recipe.dojo_class)
        if sliced_metric is not None:
            # A 17lands dojo reads a slice file, not one metric output
            family = sliced_metric.FAMILY.value
            assert key == f"seventeenlands_{family}.{sliced_metric.OUTPUT_STEM}"
            return
        metric_path = getattr(recipe, "metric_output", None) or _metric_output_path_of(
            recipe.dojo_class
        )
        assert key == f"{metric_path.parent.name}.{metric_path.stem}"

    @pytest.mark.parametrize(
        "key",
        sorted(
            k
            for k, r in DOJO_CATALOG.items()
            if isinstance(r, (CardDojoRecipe, DeckDojoRecipe)) and r.metric_output
        ),
    )
    def test_a_metric_output_override_matches_the_dojos_label_column(
        self, key: str
    ) -> None:
        # A reused wrapper reads its own metric's label column (and
        # vocabulary) from the overriding file, so they must agree
        recipe = DOJO_CATALOG[key]
        assert isinstance(recipe, (CardDojoRecipe, DeckDojoRecipe))
        own_metric = _INLINE_METRICS.get(recipe.dojo_class) or recipe.dojo_class.METRIC
        file_metric = _METRICS_BY_OUTPUT[recipe.metric_output]
        assert file_metric.LABEL_COLUMN == own_metric.LABEL_COLUMN
        assert getattr(file_metric, "LABEL_VALUES", None) == getattr(
            own_metric, "LABEL_VALUES", None
        )


class TestBuildDojos:
    def test_builds_in_order_named_by_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dojo_class = _RecordingDojoClass()
        monkeypatch.setitem(
            DOJO_CATALOG,  # type: ignore[arg-type]
            "test.one",
            CardDojoRecipe(GameId.GWENT, dojo_class),
        )
        monkeypatch.setitem(
            DOJO_CATALOG,  # type: ignore[arg-type]
            "test.two",
            CardDojoRecipe(GameId.GWENT, dojo_class),
        )

        dojos = build_dojos(["test.two", "test.one"], _context(_RecordingShelf()))

        assert [dojo.name for dojo in dojos] == ["test.two", "test.one"]

    def test_empty_names_build_nothing(self) -> None:
        assert build_dojos([], _context(_RecordingShelf())) == []

    def test_unknown_names_raise_before_loading_anything(self) -> None:
        shelf = _RecordingShelf()
        with pytest.raises(ValueError, match="nope"):
            build_dojos(["gwent_one.color_mask", "nope"], _context(shelf))
        assert shelf.requested_games == []


class TestCardShelf:
    def test_missing_deck_box_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            CardShelf().deck_box(tmp_path / "missing.db")

    def test_missing_binder_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            CardBinder, "default_output_path", lambda game: tmp_path / "missing.jsonl"
        )
        with pytest.raises(FileNotFoundError, match="gwent"):
            CardShelf().card_binder(GameId.GWENT)

    def test_loads_each_binder_once(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "gwent.jsonl"
        path.touch()
        loads: list[list[Path]] = []

        def fake_load(paths: list[Path]) -> str:
            loads.append(paths)
            return "binder"

        monkeypatch.setattr(CardBinder, "default_output_path", lambda game: path)
        monkeypatch.setattr(CardBinder, "load", fake_load)
        shelf = CardShelf()

        first = shelf.card_binder(GameId.GWENT)
        second = shelf.card_binder(GameId.GWENT)

        assert first is second
        assert loads == [[path]]


class TestContrastiveEntries:
    def test_one_per_game_with_a_deck_box(self) -> None:
        contrastive = {
            key: recipe
            for key, recipe in DOJO_CATALOG.items()
            if isinstance(recipe, ContrastiveDojoRecipe)
        }
        assert set(contrastive) == {
            "contrastive.gwent",
            "contrastive.flesh_and_blood",
            "contrastive.slay_the_spire_2",
            "contrastive.mtg",
            "contrastive.pokemon",
            "contrastive.dominion",
        }
        for key, recipe in contrastive.items():
            assert key == f"contrastive.{recipe.game.value}"

    def test_every_metric_dojo_is_a_generic_dojo(self) -> None:
        # _with_augmentations appends to a GenericDojo's pipeline
        for name, recipe in DOJO_CATALOG.items():
            if not isinstance(recipe, ContrastiveDojoRecipe):
                dojo_class = getattr(recipe, "dojo_class")
                assert issubclass(dojo_class, GenericDojo), name


class TestModOverrideValidation:
    def _context(self, overrides: dict) -> DojoBuildContext:
        return DojoBuildContext(
            shelf=_RecordingShelf(),
            holdout=HoldoutSpec.no_holdout(),
            card_embedding_size=32,
            rng_seed=0,
            mod_overrides=overrides,
        )

    def test_a_metric_dojo_takes_card_field_overrides(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recipe = CardDojoRecipe(GameId.GWENT, _RecordingDojoClass())
        monkeypatch.setitem(DOJO_CATALOG, "test.one", recipe)  # type: ignore[arg-type]
        context = self._context({"test.one": (ShuffleKeysSpec(),)})

        (dojo,) = build_dojos(["test.one"], context)

        assert isinstance(dojo, GenericDojo)
        names = [type(mod).__name__ for mod in dojo.data_mod_pipeline.mods]
        assert names == ["ShuffleKeysMod"]

    def test_an_override_for_a_dojo_not_built_is_rejected(self) -> None:
        context = self._context({"contrastive.gwent": ()})
        with pytest.raises(ValueError, match="not being built"):
            build_dojos(["gwent_one.color_mask"], context)


class TestAugmentationPipeline:
    def _context(
        self, overrides: dict | None = None, seed: int = 0
    ) -> DojoBuildContext:
        return DojoBuildContext(
            shelf=_RecordingShelf(),
            holdout=HoldoutSpec.no_holdout(),
            card_embedding_size=32,
            rng_seed=seed,
            mod_overrides=overrides or {},
        )

    def test_defaults_come_from_the_game(self) -> None:
        pipeline = _augmentation_pipeline(
            "contrastive.gwent", GameId.GWENT, self._context()
        )
        names = [type(mod).__name__ for mod in pipeline.mods]
        assert names == ["ShuffleKeysMod", "WeightedFieldMaskMod", "RandomKeyMaskMod"]

    def test_an_override_replaces_the_defaults(self) -> None:
        context = self._context({"contrastive.gwent": (ShuffleKeysSpec(),)})
        pipeline = _augmentation_pipeline("contrastive.gwent", GameId.GWENT, context)
        assert [type(mod).__name__ for mod in pipeline.mods] == ["ShuffleKeysMod"]

    def test_an_empty_override_turns_augmentation_off(self) -> None:
        context = self._context({"contrastive.gwent": ()})
        assert (
            _augmentation_pipeline("contrastive.gwent", GameId.GWENT, context).mods
            == []
        )

    def test_mod_streams_are_reproducible_and_distinct(self) -> None:
        def first_draws(name: str, seed: int) -> list[float]:
            pipeline = _augmentation_pipeline(
                name, GameId.GWENT, self._context(seed=seed)
            )
            return [mod.rng.random() for mod in pipeline.mods]  # type: ignore[attr-defined]

        draws = first_draws("contrastive.gwent", 0)
        assert draws == first_draws("contrastive.gwent", 0)
        assert len(set(draws)) == len(draws)  # no two mods share a stream
        assert draws != first_draws("contrastive.gwent", 1)
        # never the same stream as the pair constructor (seeded by rng_seed)
        assert random.Random(0).random() not in draws


class TestDeckModOptIn:
    def _context(self, overrides: dict) -> DojoBuildContext:
        return DojoBuildContext(
            shelf=_RecordingShelf(),
            holdout=HoldoutSpec.no_holdout(),
            card_embedding_size=32,
            rng_seed=0,
            mod_overrides=overrides,
        )

    def test_every_listed_dojo_is_in_the_catalog(self) -> None:
        assert set(DECK_MOD_GROUPS) <= set(DOJO_CATALOG)

    def test_no_options_group_is_ever_thinnable(self) -> None:
        for name, groups in DECK_MOD_GROUPS.items():
            dojo_class = getattr(DOJO_CATALOG[name], "dojo_class", None)
            assert dojo_class is not None, name
            assert not issubclass(dojo_class, MultiCardOptionSelectionDojo), name
            if issubclass(dojo_class, MultiGroupOptionSelectionDojo):
                assert 0 not in groups, name

    def test_every_entry_fits_its_dojos_input_shape(self) -> None:
        multi_card = (
            MultiCardBinaryClassificationDojo,
            MultiCardFixedClassificationDojo,
            MultiCardRegressionDojo,
        )
        multi_group = (
            MultiGroupBinaryClassificationDojo,
            MultiGroupOptionSelectionDojo,
            MultiGroupRegressionDojo,
        )
        for name, groups in DECK_MOD_GROUPS.items():
            dojo_class = getattr(DOJO_CATALOG[name], "dojo_class")
            if issubclass(dojo_class, multi_card):
                assert groups == frozenset({0}), name
            else:
                assert issubclass(dojo_class, multi_group), name
                assert groups and groups <= frozenset({0, 1}), name  # two groups

    @pytest.mark.parametrize(
        "name",
        [
            "sts2_runs.card_deck_size",
            "sts2_runs.total_cards_picked",
            "sts_gg.total_cards_picked",
            "isotropic.deck_card_set_copy_count",
            "isotropic.winning_deck_count",
            "final_decks.held_out_card_gwent",
            "isotropic.kingdom_opening_buy_prediction",
            "seventeenlands_draft_data.pack_to_pick_choice_set",
            "seventeenlands_replay_data.attacker_blocker_combat_outcome",
        ],
    )
    def test_count_and_answer_set_dojos_are_never_listed(self, name: str) -> None:
        assert name in DOJO_CATALOG and name not in DECK_MOD_GROUPS

    def test_a_deck_spec_is_rejected_for_an_unlisted_dojo(self) -> None:
        context = self._context({"contrastive.gwent": (CardDropoutSpec(0.1),)})
        with pytest.raises(ValueError, match="takes no deck mods"):
            build_dojos(["contrastive.gwent"], context)

    def test_a_group_the_dojo_does_not_allow_is_rejected(self) -> None:
        name = "isotropic.mid_game_next_buy"
        context = self._context({name: (CardDropoutSpec(0.1, groups=(0, 1)),)})
        with pytest.raises(ValueError, match=r"may thin groups \[1\] only"):
            build_dojos([name], context)
        assert context.shelf.requested_games == []  # type: ignore[attr-defined]


class TestMetricDojoAugmentations:
    def _context(self, overrides: dict | None = None) -> DojoBuildContext:
        return DojoBuildContext(
            shelf=_RecordingShelf(),
            holdout=HoldoutSpec.no_holdout(),
            card_embedding_size=32,
            rng_seed=0,
            mod_overrides=overrides or {},
        )

    def test_a_metric_dojo_gets_its_games_defaults_after_its_task_mods(
        self,
    ) -> None:
        task_mod = MaskTargetKeyMod("rarity", train_only=False)
        dojo = _stub_dojo("scryfall.rarity_mask", [task_mod])

        _with_augmentations(dojo, "scryfall.rarity_mask", GameId.MTG, self._context())

        mods = dojo.data_mod_pipeline.mods
        assert mods[0] is task_mod and not mods[0].train_only
        assert [type(mod).__name__ for mod in mods[1:]] == [
            "ShuffleKeysMod",
            "WeightedFieldMaskMod",
            "RandomKeyMaskMod",
        ]
        assert all(mod.train_only for mod in mods[1:])

    def test_an_override_replaces_the_defaults_after_the_task_mods(self) -> None:
        task_mod = MaskTargetKeyMod("rarity", train_only=False)
        dojo = _stub_dojo("sts2_runs.win", [task_mod])
        specs = (CardDropoutSpec(0.1), DuplicateCollapseSpec())
        context = self._context({"sts2_runs.win": specs})

        _with_augmentations(dojo, "sts2_runs.win", GameId.SLAY_THE_SPIRE_2, context)

        mods = dojo.data_mod_pipeline.mods
        assert mods[0] is task_mod
        assert [type(mod).__name__ for mod in mods[1:]] == [
            "CardDropoutMod",
            "DuplicateCollapseMod",
        ]

    def test_an_empty_override_keeps_only_the_task_mods(self) -> None:
        task_mod = MaskTargetKeyMod("rarity", train_only=False)
        dojo = _stub_dojo("scryfall.rarity_mask", [task_mod])
        context = self._context({"scryfall.rarity_mask": ()})

        _with_augmentations(dojo, "scryfall.rarity_mask", GameId.MTG, context)

        assert dojo.data_mod_pipeline.mods == [task_mod]

    def test_a_game_without_defaults_adds_nothing(self) -> None:
        dojo = _stub_dojo("x.y")
        _with_augmentations(dojo, "x.y", GameId.YUGIOH, self._context())
        assert dojo.data_mod_pipeline.mods == []


class TestStapleThresholds:
    def _context(self, thresholds: dict) -> DojoBuildContext:
        return DojoBuildContext(
            shelf=_RecordingShelf(),
            holdout=HoldoutSpec.no_holdout(),
            card_embedding_size=32,
            rng_seed=0,
            staple_thresholds=thresholds,
        )

    @pytest.mark.parametrize(
        ("names", "threshold_name"),
        [
            (["gwent_one.color_mask"], "gwent_one.color_mask"),
            (["contrastive.gwent"], "contrastive.dominion"),
        ],
    )
    def test_a_threshold_needs_a_contrastive_dojo_being_built(
        self, names: list[str], threshold_name: str
    ) -> None:
        context = self._context({threshold_name: 0.1})
        with pytest.raises(ValueError, match="not contrastive dojos being built"):
            build_dojos(names, context)
        assert context.shelf.requested_games == []  # type: ignore[attr-defined]

    def test_no_threshold_means_no_subsampling(self) -> None:
        dealer: Any = object()  # never read without a threshold
        assert (
            _staple_subsampling("contrastive.gwent", dealer, self._context({})) is None
        )


class TestRegressionObjective:
    def test_exactly_the_regression_dojo_classes_accept_an_objective(self) -> None:
        # A new regression cell outside _REGRESSION_CELLS would silently
        # train with MSE; a classifier must not be handed an objective
        for key, recipe in DOJO_CATALOG.items():
            dojo_class = getattr(recipe, "dojo_class", None)
            if dojo_class is None:
                continue
            takes = "objective" in inspect.signature(dojo_class).parameters
            assert takes == issubclass(dojo_class, _REGRESSION_CELLS), key

    def test_a_regression_class_is_bound_to_the_runs_objective(self) -> None:
        objective = RegressionObjective.huber()
        context = DojoBuildContext(
            shelf=_RecordingShelf(),
            holdout=HoldoutSpec.no_holdout(),
            card_embedding_size=32,
            rng_seed=0,
            regression_objective=objective,
        )

        bound: Any = _constructor_for(CostRegressionDojo, context)

        assert bound.keywords == {"objective": objective}

    def test_any_other_class_is_left_alone(self) -> None:
        context = DojoBuildContext(
            shelf=_RecordingShelf(),
            holdout=HoldoutSpec.no_holdout(),
            card_embedding_size=32,
            rng_seed=0,
        )
        assert _constructor_for(SetMaskDojo, context) is SetMaskDojo

    def test_the_default_objective_is_plain_mse(self) -> None:
        context = DojoBuildContext(
            shelf=_RecordingShelf(),
            holdout=HoldoutSpec.no_holdout(),
            card_embedding_size=32,
            rng_seed=0,
        )
        assert type(context.regression_objective.loss).__name__ == "MseLoss"
