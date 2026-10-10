import copy
from pathlib import Path
from typing import Any, cast

import pytest
import torch

from src.dojos.mods.card_field_mods import FieldMask
from src.dojos.mods.mod_specs import (
    CardDropoutSpec,
    CardSubsampleSpec,
    DuplicateCollapseSpec,
    RandomKeyMaskSpec,
    ShuffleKeysSpec,
    WeightedFieldMaskSpec,
)
from src.schema.game_id import GameId
from src.dojos.loss.regression_objective import RegressionLossKind
from src.training.loss_weighting import DEFAULT_BASELINE_FLOOR, LossWeighting
from src.training.plan import (
    DojoGroupRow,
    DojoRow,
    Proportional,
    SubTableRow,
    TableDiet,
    Temperature,
    Uniform,
)
from src.training.run_config import (
    ConfigDocument,
    EmbeddingHeadKind,
    GroupAttentionSpec,
    apply_overrides,
    parse_run_config,
    read_config_document,
)
from src.utils.drop_table import DropTable

_SMOKE_CONFIG = Path("configs/training/gpu_smoke.yaml")


def _document() -> ConfigDocument:
    """A small valid document; tests edit a fresh copy."""
    return {
        "run_directory": "data/runs/test",
        "device": "cpu",
        "seed": 3,
        "model": {
            "embedding_head": "linear_projection",
            "checkpoint": "ckpt",
            "card_embedding_size": 32,
        },
        "dojos": ["a", "b", "c"],
        "held_out_dojos": ["c"],
        "holdout": {"seed": 0, "tier_ratios": [8, 1, 1]},
        "hardware": {"max_batch_cost": 16, "precision": "fp16"},
        "eval_examples_per_dojo": 64,
        "phases": [
            {
                "name": "frozen",
                "diet": {"rule": "uniform"},
                "encoder_trainable": False,
                "encoder_lr": 0,
                "head_lr": 1.0e-3,
                "steps_per_round": 10,
                "max_rounds": 2,
                "saturation": {
                    "epsilon": 0.0,
                    "patience_rounds": 1,
                    "reactivation_delta": 0.1,
                    "target_saturated_fraction": 1,
                },
            }
        ],
    }


def _with(path: list[Any], value: Any) -> ConfigDocument:
    """_document() with the value at path replaced (or added)."""
    document = _document()
    container: Any = document
    for key in path[:-1]:
        container = container[key]
    container[path[-1]] = value
    return document


def _without(path: list[Any]) -> ConfigDocument:
    """_document() with the key at path removed."""
    document = _document()
    container: Any = document
    for key in path[:-1]:
        container = container[key]
    del container[path[-1]]
    return document


class TestParseRunConfig:
    def test_parses_every_section(self) -> None:
        config = parse_run_config(_document())

        assert config.run_directory == Path("data/runs/test")
        assert config.device == torch.device("cpu")
        assert config.model.embedding_head is EmbeddingHeadKind.LINEAR_PROJECTION
        assert config.model.card_embedding_size == 32
        assert config.dojo_names == ("a", "b", "c")
        assert config.plan.held_out_dojos == frozenset({"c"})
        assert config.plan.seed == 3
        assert config.plan.holdout.tier_ratios == (8.0, 1.0, 1.0)
        assert config.limits.max_batch_cost == 16
        assert config.limits.precision == "fp16"

    def test_a_phase_without_dojos_trains_every_non_held_out_dojo(self) -> None:
        phase = parse_run_config(_document()).plan.phases[0]
        assert phase.dojo_names == ("a", "b")

    def test_a_phase_can_name_its_own_dojos(self) -> None:
        config = parse_run_config(_with(["phases", 0, "dojos"], ["b"]))
        assert config.plan.phases[0].dojo_names == ("b",)

    def test_ints_are_widened_to_float(self) -> None:
        phase = parse_run_config(_document()).plan.phases[0]
        assert isinstance(phase.encoder_lr, float)
        assert isinstance(phase.saturation.target_saturated_fraction, float)

    def test_numeric_strings_are_accepted_for_floats(self) -> None:
        # PyYAML reads 1e-3 (no decimal point) as the string "1e-3"
        config = parse_run_config(_with(["phases", 0, "head_lr"], "1e-3"))
        assert config.plan.phases[0].head_lr == pytest.approx(1e-3)

    def test_absent_optional_fields_keep_their_defaults(self) -> None:
        config = parse_run_config(_document())
        assert config.plan.phases[0].max_grad_norm == 1.0
        assert config.plan.faults.max_consecutive_failures == 20

    def test_optional_fields_can_be_set(self) -> None:
        document = _with(["phases", 0, "max_grad_norm"], 0.5)
        document["faults"] = {"max_consecutive_failures": 3}
        config = parse_run_config(document)
        assert config.plan.phases[0].max_grad_norm == 0.5
        assert config.plan.faults.max_consecutive_failures == 3

    def test_held_out_games_are_parsed(self) -> None:
        document = _with(["holdout", "held_out_games"], ["gwent"])
        config = parse_run_config(document)
        assert config.plan.holdout.held_out_games == frozenset({GameId.GWENT})

    @pytest.mark.parametrize(
        ("diet", "expected"),
        [
            ({"rule": "uniform"}, Uniform()),
            ({"rule": "proportional"}, Proportional()),
            ({"rule": "temperature", "alpha": 0.5}, Temperature(alpha=0.5)),
        ],
    )
    def test_each_diet_rule(self, diet: dict[str, Any], expected: object) -> None:
        config = parse_run_config(_with(["phases", 0, "diet"], diet))
        assert config.plan.phases[0].diet_rule == expected

    @pytest.mark.parametrize(
        ("path", "value", "message"),
        [
            (["typo"], 1, "config.typo"),
            (["model", "typo"], 1, "config.model.typo"),
            (["hardware", "typo"], 1, "config.hardware.typo"),
            (["holdout", "typo"], 1, "config.holdout.typo"),
            (["phases", 0, "typo"], 1, "config.phases.0.typo"),
            (["phases", 0, "diet", "typo"], 1, "config.phases.0.diet.typo"),
            (["phases", 0, "saturation", "typo"], 1, "phases.0.saturation.typo"),
        ],
    )
    def test_unknown_keys_raise_at_any_depth(
        self, path: list[Any], value: Any, message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            parse_run_config(_with(path, value))

    def test_faults_typo_raises(self) -> None:
        document = _document()
        document["faults"] = {"max_consecutive_failure": 3}
        with pytest.raises(ValueError, match="config.faults.max_consecutive_failure"):
            parse_run_config(document)

    @pytest.mark.parametrize(
        "path",
        [
            ["device"],
            ["model", "card_embedding_size"],
            ["phases", 0, "head_lr"],
            ["hardware"],
        ],
    )
    def test_missing_required_keys_raise(self, path: list[Any]) -> None:
        with pytest.raises(ValueError, match="is required"):
            parse_run_config(_without(path))

    @pytest.mark.parametrize(
        ("path", "value"),
        [
            (["seed"], True),  # bool is not an int
            (["seed"], "3"),
            (["phases", 0, "head_lr"], "fast"),
            (["phases", 0, "head_lr"], "nan"),
            (["phases", 0, "head_lr"], "inf"),
            (["phases", 0, "head_lr"], float("inf")),  # YAML-native .inf
            (["phases", 0, "head_lr"], 10**400),  # overflows float()
            (["phases", 0, "encoder_trainable"], 0),
            (["dojos"], ["a", 1]),
            (["dojos"], "a"),
            (["holdout", "tier_ratios"], [8, 1]),
            (["phases"], [1]),
        ],
    )
    def test_wrongly_typed_values_raise(self, path: list[Any], value: Any) -> None:
        with pytest.raises(ValueError):
            parse_run_config(_with(path, value))

    @pytest.mark.parametrize(
        ("path", "value", "message"),
        [
            (["model", "embedding_head"], "transformer", "config.model.embedding_head"),
            (["phases", 0, "diet"], {"rule": "greedy"}, "diet.rule"),
            (["holdout", "held_out_games"], ["chess"], "unknown game"),
            (["device"], "not_a_device", "config.device"),
            (["model", "card_embedding_size"], 0, "card_embedding_size"),
            (["phases", 0, "max_rounds"], 0, "config.phases.0"),
        ],
    )
    def test_invalid_values_raise_with_their_location(
        self, path: list[Any], value: Any, message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            parse_run_config(_with(path, value))

    def test_temperature_needs_alpha(self) -> None:
        with pytest.raises(ValueError, match="alpha is required"):
            parse_run_config(_with(["phases", 0, "diet"], {"rule": "temperature"}))

    def test_alpha_on_uniform_is_rejected(self) -> None:
        diet = {"rule": "uniform", "alpha": 0.5}
        with pytest.raises(ValueError, match="diet.alpha"):
            parse_run_config(_with(["phases", 0, "diet"], diet))

    def test_duplicate_dojos_raise(self) -> None:
        with pytest.raises(ValueError, match="more than once"):
            parse_run_config(_with(["dojos"], ["a", "b", "a", "c"]))

    def test_a_phase_naming_an_unlisted_dojo_raises(self) -> None:
        with pytest.raises(ValueError, match="not in dojos"):
            parse_run_config(_with(["phases", 0, "dojos"], ["z"]))

    def test_an_unlisted_held_out_dojo_joins_the_run(self) -> None:
        config = parse_run_config(_with(["held_out_dojos"], ["z"]))
        assert config.dojo_names == ("a", "b", "c", "z")
        assert config.plan.held_out_dojos == frozenset({"z"})

    def test_a_phase_training_a_held_out_dojo_raises(self) -> None:
        with pytest.raises(ValueError, match="held-out"):
            parse_run_config(_with(["phases", 0, "dojos"], ["c"]))

    def test_the_committed_smoke_config_parses(self) -> None:
        config = parse_run_config(read_config_document(_SMOKE_CONFIG))
        assert config.limits.precision == "fp16"
        assert len(config.plan.phases[0].dojo_names) == len(config.dojo_names)


class TestDojoPatterns:
    _CATALOG = (
        "contrastive.mtg",
        "contrastive.gwent",
        "contrastive_odd_one_out.mtg",
        "sts2_runs.card_win_rate",
        "sts2_runs.card_win_rate_at_act2",
        "scryfall.cmc_regression",
    )

    def _parse(self, dojos: list[str], held_out: list[str]) -> Any:
        document = _with(["dojos"], dojos)
        document["held_out_dojos"] = held_out
        return parse_run_config(document, self._CATALOG)

    def test_a_pattern_adds_catalog_matches_in_catalog_order(self) -> None:
        config = self._parse(["a", "contrastive*.*", "c"], ["c"])
        assert config.dojo_names == (
            "a",
            "contrastive.mtg",
            "contrastive.gwent",
            "contrastive_odd_one_out.mtg",
            "c",
        )

    def test_an_exclusion_drops_earlier_matches(self) -> None:
        config = self._parse(["contrastive*.*", "!*.gwent", "c"], ["c"])
        assert config.dojo_names == (
            "contrastive.mtg",
            "contrastive_odd_one_out.mtg",
            "c",
        )

    def test_overlapping_patterns_add_each_name_once(self) -> None:
        config = self._parse(["contrastive.*", "contrastive.mtg", "*.mtg"], [])
        assert config.dojo_names.count("contrastive.mtg") == 1

    def test_held_out_patterns_join_the_run_and_are_held_out(self) -> None:
        config = self._parse(["contrastive.mtg"], ["sts2_runs.card_win_rate*"])
        held_out = ("sts2_runs.card_win_rate", "sts2_runs.card_win_rate_at_act2")
        assert config.dojo_names == ("contrastive.mtg", *held_out)
        assert config.plan.held_out_dojos == frozenset(held_out)
        assert config.plan.phases[0].dojo_names == ("contrastive.mtg",)

    def test_a_held_out_dojo_also_in_dojos_is_listed_once(self) -> None:
        config = self._parse(["*.cmc_regression", "contrastive.mtg"], ["scryfall.*"])
        assert config.dojo_names == ("scryfall.cmc_regression", "contrastive.mtg")

    def test_a_pattern_matching_nothing_raises(self) -> None:
        with pytest.raises(ValueError, match=r"dojos\.1: 'hearthstone\.\*' matches"):
            self._parse(["a", "hearthstone.*"], [])

    def test_an_exclusion_matching_nothing_raises(self) -> None:
        with pytest.raises(ValueError, match="excludes nothing"):
            self._parse(["contrastive.*", "!*.pokemon"], [])

    def test_without_a_catalog_a_pattern_matches_nothing(self) -> None:
        with pytest.raises(ValueError, match="matches no dojo"):
            parse_run_config(_with(["dojos"], ["a*", "c"]))


class TestReadConfigDocument:
    def test_rejects_a_non_mapping_top_level(self, tmp_path: Path) -> None:
        path = tmp_path / "config.yaml"
        path.write_text("- a\n- b\n", encoding="utf-8")
        with pytest.raises(ValueError, match="mapping"):
            read_config_document(path)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            read_config_document(tmp_path / "missing.yaml")


class TestApplyOverrides:
    def test_sets_nested_values_through_list_indexes(self) -> None:
        document = apply_overrides(
            _document(), ["phases.0.max_rounds=7", "hardware.precision=fp32"]
        )
        assert document["phases"][0]["max_rounds"] == 7
        assert document["hardware"]["precision"] == "fp32"

    def test_values_are_parsed_as_yaml(self) -> None:
        document = apply_overrides(_document(), ["dojos=[a, b]", "seed=5"])
        assert document["dojos"] == ["a", "b"]
        assert document["seed"] == 5

    def test_can_add_a_new_final_key(self) -> None:
        document = apply_overrides(_document(), ["phases.0.max_grad_norm=0.5"])
        assert document["phases"][0]["max_grad_norm"] == 0.5

    def test_later_overrides_win(self) -> None:
        document = apply_overrides(_document(), ["seed=1", "seed=2"])
        assert document["seed"] == 2

    def test_does_not_modify_its_input(self) -> None:
        original = _document()
        before = copy.deepcopy(original)
        apply_overrides(original, ["seed=9", "phases.0.max_rounds=9"])
        assert original == before

    def test_no_overrides_is_a_copy(self) -> None:
        assert apply_overrides(_document(), []) == _document()

    def test_a_bad_final_index_names_the_whole_path(self) -> None:
        with pytest.raises(ValueError, match="phases.3"):
            apply_overrides(_document(), ["phases.3=x"])

    @pytest.mark.parametrize(
        "override",
        [
            "seed",  # no "="
            "missing.key=1",  # intermediate key does not exist
            "phases.5.max_rounds=1",  # index out of range
            "phases.x.max_rounds=1",  # not an index
            "seed.inner=1",  # walks through a scalar
            "phases..name=x",  # empty segment
            "phases.3=x",  # final index out of range
        ],
    )
    def test_bad_overrides_raise(self, override: str) -> None:
        with pytest.raises(ValueError):
            apply_overrides(_document(), [override])


def _weighted(table: Any) -> dict:
    return {"kind": "weighted_field_mask", "table": table}


class TestModOverrides:
    def _with_mods(self, mods: Any) -> ConfigDocument:
        document = _document()
        document["mods"] = mods
        return document

    def test_parses_every_kind(self) -> None:
        nested = [
            {"weight": 1, "mask": [["back_face", "name"]]},
            {"weight": 1, "mask": [["card_faces", 0, "name"]]},
        ]
        mods = {
            "a": [
                {"kind": "shuffle_keys"},
                {"kind": "random_key_mask", "probability": 0.1},
                _weighted(
                    [
                        {"weight": 40},
                        {"weight": 35, "mask": [["faction"], ["faction-duo"]]},
                        {"weight": 10, "table": nested},
                    ]
                ),
            ]
        }
        specs = parse_run_config(self._with_mods(mods)).mod_overrides["a"]

        assert isinstance(specs[0], ShuffleKeysSpec)
        assert specs[1] == RandomKeyMaskSpec(probability=0.1)
        table = cast(WeightedFieldMaskSpec, specs[2]).table
        assert table.entries[0].outcome == FieldMask()
        assert table.entries[1].outcome == FieldMask.of_keys("faction", "faction-duo")
        sub = table.entries[2].outcome
        assert isinstance(sub, DropTable)
        assert sub.entries[1].outcome == FieldMask((("card_faces", 0, "name"),))

    def test_parses_every_deck_kind(self) -> None:
        mods = {
            "a": [
                {"kind": "card_dropout", "drop_probability": 0.1},
                {"kind": "card_subsample", "keep_fraction": 0.8, "groups": [1]},
                {"kind": "duplicate_collapse", "groups": [0, 1]},
            ]
        }
        specs = parse_run_config(self._with_mods(mods)).mod_overrides["a"]

        assert specs == (
            CardDropoutSpec(drop_probability=0.1, groups=(0,)),
            CardSubsampleSpec(keep_fraction=0.8, groups=(1,)),
            DuplicateCollapseSpec(groups=(0, 1)),
        )

    @pytest.mark.parametrize(
        ("entry", "message"),
        [
            ({"kind": "card_dropout", "drop_probability": 2}, "config.mods.a.0"),
            ({"kind": "card_subsample"}, "keep_fraction is required"),
            ({"kind": "duplicate_collapse", "groups": 1}, "must be a list"),
            ({"kind": "duplicate_collapse", "groups": ["x"]}, r"groups\.0"),
            ({"kind": "duplicate_collapse", "groups": [-1]}, "config.mods.a.0"),
        ],
    )
    def test_bad_deck_mods_raise_with_their_location(
        self, entry: Any, message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            parse_run_config(self._with_mods({"a": [entry]}))

    def test_parses_staple_thresholds(self) -> None:
        document = _document()
        document["staple_subsampling"] = {"a": 0.1}
        assert parse_run_config(document).staple_thresholds == {"a": 0.1}
        assert parse_run_config(_document()).staple_thresholds == {}

    @pytest.mark.parametrize(
        ("thresholds", "message"),
        [
            ({"z": 0.1}, "not in dojos"),
            ({"a": 0}, "must be > 0"),
            ({"a": -1}, "must be > 0"),
            ({"a": float("inf")}, "finite"),
            ({"a": "x"}, "number"),
        ],
    )
    def test_bad_staple_thresholds_raise(self, thresholds: Any, message: str) -> None:
        document = _document()
        document["staple_subsampling"] = thresholds
        with pytest.raises(ValueError, match=message):
            parse_run_config(document)

    def test_parses_loss_weights_and_the_floor(self) -> None:
        document = _document()
        document["loss_weights"] = {"a": 2.0, "b": 0.75}
        document["baseline_floor"] = 0.1
        weighting = parse_run_config(document).plan.loss_weighting
        assert weighting is not None
        assert dict(weighting.weights) == {"a": 2.0, "b": 0.75}
        assert weighting.baseline_floor == 0.1

    def test_absent_loss_weights_still_normalize_every_dojo(self) -> None:
        weighting = parse_run_config(_document()).plan.loss_weighting
        assert weighting == LossWeighting({}, DEFAULT_BASELINE_FLOOR)

    @pytest.mark.parametrize(
        ("key", "value", "message"),
        [
            ("loss_weights", {"a": 0}, r"loss_weights\.a must be finite and > 0"),
            ("loss_weights", {"a": -1.0}, "must be finite and > 0"),
            ("loss_weights", {"a": "x"}, "number"),
            ("loss_weights", {"c": 2.0}, "loss_weights names"),
            ("loss_weights", {"zzz": 2.0}, "loss_weights names"),
            ("baseline_floor", 0, r"baseline_floor must be finite and > 0"),
            ("baseline_floor", "x", "number"),
        ],
    )
    def test_bad_loss_weighting_raises(
        self, key: str, value: Any, message: str
    ) -> None:
        document = _document()
        document[key] = value
        with pytest.raises(ValueError, match=message):
            parse_run_config(document)

    def test_weight_by_baseline_false_trains_on_the_raw_loss(self) -> None:
        document = _document()
        document["weight_by_baseline"] = False
        assert parse_run_config(document).plan.loss_weighting is None

    @pytest.mark.parametrize(
        ("key", "value"), [("loss_weights", {"a": 2.0}), ("baseline_floor", 0.1)]
    )
    def test_weights_alongside_weight_by_baseline_false_raise(
        self, key: str, value: Any
    ) -> None:
        document = _document()
        document["weight_by_baseline"] = False
        document[key] = value
        with pytest.raises(ValueError, match="weight_by_baseline"):
            parse_run_config(document)

    def test_regression_loss_defaults_to_mse(self) -> None:
        assert parse_run_config(_document()).regression_loss is RegressionLossKind.MSE

    @pytest.mark.parametrize("name", ["mse", "huber"])
    def test_parses_regression_loss(self, name: str) -> None:
        document = _document()
        document["regression_loss"] = name
        assert parse_run_config(document).regression_loss is RegressionLossKind(name)

    def test_a_bad_regression_loss_raises_with_its_location(self) -> None:
        document = _document()
        document["regression_loss"] = "l1"
        with pytest.raises(ValueError, match=r"config\.regression_loss.*'l1'"):
            parse_run_config(document)

    def test_an_empty_list_means_no_augmentation(self) -> None:
        assert parse_run_config(self._with_mods({"a": []})).mod_overrides == {"a": ()}

    def test_absent_mods_means_no_overrides(self) -> None:
        assert parse_run_config(_document()).mod_overrides == {}

    @pytest.mark.parametrize(
        ("mods", "message"),
        [
            ({"z": []}, "not in dojos"),
            ({"a": [{"kind": "jitter"}]}, "config.mods.a.0.kind"),
            ({"a": [{"kind": "shuffle_keys", "typo": 1}]}, "config.mods.a.0.typo"),
            ({"a": [{"kind": "random_key_mask", "probability": 2}]}, "config.mods.a.0"),
            ({"a": [_weighted([])]}, "config.mods.a.0.table"),
            (
                {"a": [_weighted([{"weight": 1, "mask": [["x"]], "table": []}])]},
                "both mask and table",
            ),
            ({"a": [_weighted([{"weight": 1, "mask": ["x"]}])]}, "list of steps"),
            ({"a": [_weighted([{"weight": 1, "mask": [["x", 1.5]]}])]}, "str or int"),
            (
                {"a": [_weighted([{"weight": 1, "mask": [["x"], ["x", "y"]]}])]},
                "overlap",
            ),
        ],
    )
    def test_bad_mods_raise_with_their_location(self, mods: Any, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            parse_run_config(self._with_mods(mods))


class TestTableDiet:
    """phases[0].diet as {rule: table, table: [...]}; run dojos are a, b
    (c is held out)."""

    @staticmethod
    def _with_table(rows: list[Any]) -> ConfigDocument:
        return _with(["phases", 0, "diet"], {"rule": "table", "table": rows})

    def test_rows_of_every_kind_parse(self) -> None:
        document = self._with_table(
            [
                {"weight": 1, "dojo": "a"},
                {
                    "weight": 2,
                    "table": [
                        {
                            "weight": 1,
                            "dojos": ["b*"],
                            "within": {"rule": "temperature", "alpha": 0.3},
                        }
                    ],
                },
            ]
        )
        phase = parse_run_config(document).plan.phases[0]
        assert phase.diet_rule == TableDiet(
            (
                DojoRow(1.0, "a"),
                SubTableRow(2.0, (DojoGroupRow(1.0, ("b",), Temperature(0.3)),)),
            )
        )
        assert phase.dojo_names == ("a", "b")

    def test_within_defaults_to_uniform_and_patterns_keep_run_order(self) -> None:
        document = self._with_table([{"weight": 1, "dojos": ["b", "?"]}])
        diet = parse_run_config(document).plan.phases[0].diet_rule
        assert diet == TableDiet((DojoGroupRow(1.0, ("a", "b"), Uniform()),))

    @pytest.mark.parametrize(
        "rows, message",
        [
            ([{"weight": 1, "dojos": ["z*"]}], "match no dojo"),
            ([{"weight": 1, "dojos": ["c"]}], "match no dojo"),
            ([{"weight": 1}], "exactly one of"),
            ([{"weight": 1, "dojo": "a", "dojos": ["b"]}], "exactly one of"),
            (
                [{"weight": 1, "dojo": "a"}, {"weight": 1, "dojos": ["*"]}],
                "more than once",
            ),
            ([{"weight": 0, "dojo": "a"}], "positive weight"),
            ([{"weight": 1, "dojo": "a", "extra": 1}], "unknown config keys"),
            (
                [{"weight": 1, "dojos": ["a"], "within": {"rule": "table"}}],
                "want one of",
            ),
            ([{"weight": 1, "dojo": "c"}], "held-out"),
            ([{"weight": 1, "dojo": "zzz"}], "not in dojos"),
        ],
    )
    def test_bad_tables_raise(self, rows: list[Any], message: str) -> None:
        with pytest.raises(ValueError, match=message):
            parse_run_config(self._with_table(rows))

    def test_a_table_phase_cannot_also_list_dojos(self) -> None:
        document = self._with_table([{"weight": 1, "dojo": "a"}])
        document["phases"][0]["dojos"] = ["a"]
        with pytest.raises(ValueError, match="names the phase's dojos itself"):
            parse_run_config(document)

    def test_an_unknown_phase_rule_suggests_table(self) -> None:
        document = _with(["phases", 0, "diet"], {"rule": "tabel"})
        with pytest.raises(ValueError, match="'table'"):
            parse_run_config(document)


class TestModelSpec:
    def test_no_group_attention_is_a_single_card_model(self) -> None:
        config = parse_run_config(_document())
        assert config.model.group_attention is None

    def test_an_empty_group_attention_takes_the_defaults(self) -> None:
        config = parse_run_config(_with(["model", "group_attention"], {}))
        assert config.model.group_attention == GroupAttentionSpec()

    def test_group_attention_fields_are_parsed(self) -> None:
        attention = {
            "num_layers": 3,
            "num_heads": 8,
            "ffn_dim": 64,
            "dropout": 0,
            "norm_first": False,
        }
        config = parse_run_config(_with(["model", "group_attention"], attention))
        assert config.model.group_attention == GroupAttentionSpec(3, 8, 64, 0.0, False)

    def test_residual_mlp_sizes_are_parsed(self) -> None:
        document = _document()
        document["model"] |= {
            "embedding_head": "residual_mlp",
            "mlp_hidden_dim": 64,
            "mlp_num_blocks": 7,
        }
        model = parse_run_config(document).model
        assert (model.mlp_hidden_dim, model.mlp_num_blocks) == (64, 7)

    def test_mlp_sizes_on_another_head_raise(self) -> None:
        with pytest.raises(ValueError, match="only to embedding_head: residual_mlp"):
            parse_run_config(_with(["model", "mlp_num_blocks"], 7))

    @pytest.mark.parametrize(
        ("attention", "message"),
        [
            ({"num_heads": 5}, "must divide card_embedding_size"),
            ({"num_layers": 0}, "num_layers must be >= 1"),
            ({"dropout": 1}, "dropout"),
            ({"typo": 1}, "config.model.group_attention.typo"),
        ],
    )
    def test_bad_group_attention_raises(
        self, attention: dict[str, Any], message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            parse_run_config(_with(["model", "group_attention"], attention))
