import random
from pathlib import Path
from typing import Any

import pytest

from src.data_refinement.card_binder.card_binder import CardBinder
from src.dojos.mods.mod_specs import ShuffleKeysSpec
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
from src.data_refinement.metrics.sts_gg.deck_label_metrics import (
    CharacterPredictionMetric,
    WinMetric,
)
from src.dojos import isotropic
from src.dojos.dominiontabs.cost_regression_dojo import CostRegressionDojo
from src.dojos.dominiontabs.masked_field_dojos import SetMaskDojo
from src.dojos.play_gwent.deck_card_mask_dojos import LeaderMaskedFromDeckDojo
from src.dojos.sts_gg import deck_label_dojos
from src.schema.game_id import GameId
from src.schema.holdout import HoldoutSpec
from src.training.dojo_catalog import (
    DOJO_CATALOG,
    CardDojoRecipe,
    CardShelf,
    ContrastiveDojoRecipe,
    DeckDojoRecipe,
    DojoBuildContext,
    _augmentation_pipeline,
    build_dojos,
    takes_augmentations,
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

    def deck_box(self, path: Path) -> Any:
        self.requested_boxes.append(path)
        return f"box:{path.name}"


class _RecordingDojoClass:
    """Stands in for a dojo class; records the arguments it was built with."""

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append((args, kwargs))
        return kwargs["name"]


# Dojos that name their metric in their constructor instead of a METRIC ClassVar
_INLINE_METRICS: dict[Any, Any] = {
    deck_label_dojos.WinDojo: WinMetric,
    deck_label_dojos.CharacterDojo: CharacterPredictionMetric,
    CostRegressionDojo: CostRegressionMetric,
    SetMaskDojo: DominionSetMaskMetric,
    LeaderMaskedFromDeckDojo: LeaderMaskedFromDeckMetric,
    isotropic.FullDeckWinPredictionDojo: FullDeckWinPredictionMetric,
    isotropic.KingdomEndingTypeDojo: KingdomEndingTypeMetric,
    isotropic.WinningDeckMaskedCardDojo: WinningDeckMaskedCardMetric,
}


def _metric_output_path_of(dojo_class: Any) -> Path:
    if hasattr(dojo_class, "OUTPUT_PATH"):  # isotropic dojos name the path
        return dojo_class.OUTPUT_PATH
    metric = _INLINE_METRICS.get(dojo_class) or dojo_class.METRIC
    return metric.DEFAULT_OUTPUT_PATH


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
                {"name": "gwent_one.x", "rng_seed": 7},
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
        assert kwargs == {"name": "sts_gg.win", "rng_seed": 7}
        assert shelf.requested_boxes == [Path("x/box.db")]


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
        assert isinstance(recipe, (CardDojoRecipe, DeckDojoRecipe))
        metric_path = _metric_output_path_of(recipe.dojo_class)
        assert key == f"{metric_path.parent.name}.{metric_path.stem}"


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

        assert dojos == ["test.two", "test.one"]

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

    def test_only_contrastive_dojos_take_augmentations(self) -> None:
        assert takes_augmentations("contrastive.gwent")
        assert not takes_augmentations("gwent_one.color_mask")


class TestModOverrideValidation:
    def _context(self, overrides: dict) -> DojoBuildContext:
        return DojoBuildContext(
            shelf=_RecordingShelf(),
            holdout=HoldoutSpec.no_holdout(),
            card_embedding_size=32,
            rng_seed=0,
            mod_overrides=overrides,
        )

    def test_an_override_for_a_metric_dojo_is_rejected(self) -> None:
        context = self._context({"gwent_one.color_mask": ()})
        with pytest.raises(ValueError, match="take no augmentation"):
            build_dojos(["gwent_one.color_mask"], context)
        assert context.shelf.requested_games == []  # type: ignore[attr-defined]

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
