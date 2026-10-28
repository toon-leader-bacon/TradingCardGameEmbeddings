"""A training run's configuration file: read, override, parse.

The file (YAML) is the edge: `read_config_document` returns its raw
mapping, `apply_overrides` edits that mapping from `--set key.path=value`
strings, and `parse_run_config` turns it into the frozen types the trainer
already uses (TrainingPlan, Phase, HoldoutSpec, HardwareLimits) plus a
ModelSpec. Nothing past parse_run_config sees the raw mapping, except the
copy the driver writes into the run directory as the record of what ran
(defaults the file left out, e.g. FaultPolicy, are not in that copy).

Unknown keys raise, so a typo fails at startup instead of silently falling
back to a default. Every error names its location ("phases.0.saturation").

This module parses the top level, phases and holdout; the larger
sections have their own parsers in config_parsing/ (diet, model, mods),
built on config_parsing/config_section.py's ConfigSection. The model
specs themselves live in model_spec.py, beside plan.py.
"""

import copy
import fnmatch
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import yaml

from src.dojos.loss.regression_objective import RegressionLossKind
from src.dojos.mods.mod_specs import AugmentationRates, ModSpec
from src.schema.game_id import GameId
from src.schema.holdout import DEFAULT_DECK_SPLIT_RATIOS, HoldoutSpec
from src.training.config_parsing.config_section import (
    ConfigSection,
    build_flat_dataclass,
    checked_scalar,
    construct_at,
)
from src.training.config_parsing.diet_parsing import parse_diet
from src.training.config_parsing.mod_parsing import (
    parse_default_augmentation,
    parse_mod_overrides,
)
from src.training.config_parsing.model_parsing import parse_model
from src.training.learning_rate_schedule import LearningRateSchedule
from src.training.loss_weighting import DEFAULT_BASELINE_FLOOR, LossWeighting
from src.training.model_spec import ModelSpec
from src.training.plan import (
    DietRule,
    FaultPolicy,
    HardwareLimits,
    Phase,
    SaturationSpec,
    TableDiet,
    TrainingPlan,
)

# The raw, untyped YAML mapping. Exists only between the file and
# parse_run_config (and in the run directory's written copy).
ConfigDocument = dict[str, Any]

_TIER_COUNT = 3


@dataclass(frozen=True)
class RunConfig:
    """Everything one training run needs, parsed and validated.

    run_directory: where checkpoints, CSV logs and the config copy go;
        the driver requires that it not exist yet.
    device: where the model and dojo heads run (ROCm also uses "cuda").
    model: the encoder to build.
    dojo_names: every dojo to construct, as DOJO_CATALOG keys, no
        duplicates. Diet dojos and held-out dojos are both drawn from it.
    plan: the TrainingPlan (phases, holdout, held-out dojos, seed).
    limits: HardwareLimits (batch cost ceiling, precision).
    mod_overrides: per run dojo, augmentation specs replacing that dojo's
        game defaults (empty tuple: no augmentation); a metric dojo's own
        task mods stay and run first. Checked here to name run dojos;
        build_dojos checks deck specs against dojo_catalog.DECK_MOD_GROUPS.
    staple_thresholds: per run dojo, the staple-subsampling t (finite, > 0)
        of a contrastive dojo (see src/dojos/contrastive/
        staple_subsampling.py); a dojo not named keeps t = inf (no
        subsampling). Checked here to name run dojos; build_dojos checks
        each is contrastive.
    regression_loss: the loss every regression dojo trains with (mse,
        today's, or huber); the run's DojoBuildContext carries the matching
        RegressionObjective.
    warm_feature_cache: games whose every card is encoded into the model's
        feature cache before training (top-level `warm_feature_cache:
        [gwent, mtg]`); empty by default. Needs model.feature_cache.
    default_augmentation: rates rewriting every game default card-field
        mod (top-level `default_augmentation: {probability, variants}`);
        dojos named in mod_overrides are not touched. Nothing by default.
    """

    run_directory: Path
    device: torch.device
    model: ModelSpec
    dojo_names: tuple[str, ...]
    plan: TrainingPlan
    limits: HardwareLimits
    mod_overrides: Mapping[str, tuple[ModSpec, ...]] = field(default_factory=dict)
    staple_thresholds: Mapping[str, float] = field(default_factory=dict)
    regression_loss: RegressionLossKind = RegressionLossKind.MSE
    warm_feature_cache: tuple[GameId, ...] = ()
    default_augmentation: AugmentationRates = AugmentationRates()

    def __post_init__(self) -> None:
        # No duplicate dojos (the Trainer keys dojos by name and would
        # silently keep one), and every phase and held-out dojo is a run dojo
        duplicates = sorted(
            {name for name in self.dojo_names if self.dojo_names.count(name) > 1}
        )
        if duplicates:
            raise ValueError(f"dojos lists {duplicates} more than once")
        known = set(self.dojo_names)
        unknown_held_out = sorted(self.plan.held_out_dojos - known)
        if unknown_held_out:
            raise ValueError(f"held_out_dojos {unknown_held_out} are not in dojos")
        for phase in self.plan.phases:
            unknown = sorted(set(phase.dojo_names) - known)
            if unknown:
                raise ValueError(f"phase {phase.name!r} names {unknown}, not in dojos")
        unknown_overrides = sorted(set(self.mod_overrides) - known)
        if unknown_overrides:
            raise ValueError(f"mods names {unknown_overrides}, not in dojos")
        unknown_thresholds = sorted(set(self.staple_thresholds) - known)
        if unknown_thresholds:
            raise ValueError(
                f"staple_subsampling names {unknown_thresholds}, not in dojos"
            )
        # Warming needs a cache to warm
        if self.warm_feature_cache and self.model.feature_cache is None:
            raise ValueError("warm_feature_cache needs a model.feature_cache section")


def read_config_document(path: Path) -> ConfigDocument:
    """Read a YAML run config into its raw mapping.

    Inputs: path (Path) to a YAML file whose top level is a mapping.
    Output: ConfigDocument.
    Side effects: reads the file.
    Exceptions: FileNotFoundError; ValueError if the top level is not a
        mapping; yaml.YAMLError on malformed YAML.

    Example:
        >>> document = read_config_document(Path("configs/training/gpu_smoke.yaml"))
    """
    with path.open(encoding="utf-8") as config_file:
        raw = yaml.safe_load(config_file)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: the top level must be a mapping")
    return raw


def apply_overrides(
    document: ConfigDocument, overrides: Sequence[str]
) -> ConfigDocument:
    """Return a copy of document with each "dotted.key=value" override set.

    A path segment that is an integer indexes a list (phases.0.max_rounds).
    The value is parsed as YAML, so "3" is an int, "fp16" a string and
    "[a, b]" a list. The final key may be new (to set an optional field);
    every intermediate key must already exist.

    Inputs: document (ConfigDocument, not modified), overrides (Sequence[str]).
    Output: a new ConfigDocument.
    Side effects: none.
    Exceptions: ValueError for an override with no "=", or whose path
        walks through a missing key, a non-container, or a list index out
        of range.

    Example:
        >>> apply_overrides(document, ["phases.0.max_rounds=3", "hardware.precision=fp32"])
    """
    result = copy.deepcopy(document)
    # Apply each override in order, so a later one wins
    for override in overrides:
        path, value = _split_override(override)
        container, last_key = _walk_to_parent(result, path)
        _set_in_container(container, last_key, value, ".".join(path))
    return result


def parse_run_config(
    document: ConfigDocument, catalog_names: Sequence[str] = ()
) -> RunConfig:
    """Parse a raw config document into a validated RunConfig.

    "dojos" and "held_out_dojos" entries are dojo names or fnmatch
    patterns ("sts2_runs.*"), expanded against catalog_names
    (_expanded_dojo_entries); a "dojos" entry "!pattern" (quoted in YAML)
    drops the names matched so far. Held-out dojos are run dojos too:
    any not already in "dojos" are appended to the run's dojo_names.

    Expected top-level keys: run_directory, device, seed, model, dojos,
    held_out_dojos (optional), holdout, hardware, eval_examples_per_dojo,
    phases, faults (optional), mods (optional: dojo name -> list of mod
    entries, see config_parsing/mod_parsing.py), default_augmentation
    (optional: {probability, variants}, see
    mod_parsing.parse_default_augmentation), staple_subsampling (optional: dojo name
    -> t, see _parse_staple_thresholds), loss_weights (optional: dojo name
    -> multiplier), baseline_floor and weight_by_baseline (optional; see
    _parse_loss_weighting), and regression_loss (optional: mse or huber;
    see _parse_regression_loss).
    A phase without a "dojos" key trains every run dojo that is not held out.

    Inputs: document (ConfigDocument), catalog_names (every dojo name a
        pattern may match, in the order matches are listed; the driver
        passes DOJO_CATALOG's keys; empty means patterns match nothing).
    Output: RunConfig.
    Side effects: none.
    Exceptions: ValueError for a missing or unknown key, a wrongly typed
        value, a pattern matching no dojo, or anything the frozen types'
        own validation rejects.

    Example:
        >>> config = parse_run_config(read_config_document(path), tuple(DOJO_CATALOG))
        >>> config.plan.phases[0].name
        'frozen'
    """
    top = ConfigSection(document, "config")
    listed_names = _expanded_dojo_entries(
        top.required_str_list("dojos"), catalog_names, top.location_of("dojos")
    )
    held_out_names = _expanded_dojo_entries(
        top.optional_str_list("held_out_dojos"),
        catalog_names,
        top.location_of("held_out_dojos"),
    )
    dojo_names = listed_names + tuple(
        name for name in held_out_names if name not in listed_names
    )
    held_out = frozenset(held_out_names)
    trainable_names = tuple(name for name in dojo_names if name not in held_out)

    # Build the plan from its parts
    phases = tuple(
        _parse_phase(section, trainable_names)
        for section in top.required_section_list("phases")
    )
    holdout = _parse_holdout(top.required_section("holdout"))
    eval_examples_per_dojo = top.required("eval_examples_per_dojo", int)
    seed = top.required("seed", int)
    faults = build_flat_dataclass(FaultPolicy, top.optional_section("faults"))
    loss_weighting = _parse_loss_weighting(top)
    plan = construct_at(
        "config",
        lambda: TrainingPlan(
            phases=phases,
            holdout=holdout,
            held_out_dojos=held_out,
            eval_examples_per_dojo=eval_examples_per_dojo,
            seed=seed,
            faults=faults,
            loss_weighting=loss_weighting,
        ),
    )

    # Then everything around it
    run_directory = Path(top.required("run_directory", str))
    device = _parse_device(top.required("device", str), top.location_of("device"))
    model = parse_model(top.required_section("model"))
    limits = build_flat_dataclass(HardwareLimits, top.required_section("hardware"))
    regression_loss = _parse_regression_loss(top)
    mod_overrides = parse_mod_overrides(top.optional_section("mods"))
    default_augmentation = parse_default_augmentation(
        top.optional_section("default_augmentation")
    )
    staple_thresholds = _parse_staple_thresholds(
        top.optional_section("staple_subsampling")
    )
    warm_feature_cache = tuple(
        _parse_game(value, top.location_of("warm_feature_cache"))
        for value in top.optional_str_list("warm_feature_cache")
    )
    top.reject_unread_keys()
    result = construct_at(
        "config",
        lambda: RunConfig(
            run_directory=run_directory,
            device=device,
            model=model,
            dojo_names=dojo_names,
            plan=plan,
            limits=limits,
            mod_overrides=mod_overrides,
            staple_thresholds=staple_thresholds,
            regression_loss=regression_loss,
            warm_feature_cache=warm_feature_cache,
            default_augmentation=default_augmentation,
        ),
    )
    return result


def _parse_phase(section: ConfigSection, default_dojos: tuple[str, ...]) -> Phase:
    """One phases[] entry -> Phase; "dojos" defaults to default_dojos. A
    table diet names the phase's dojos itself, so "dojos" is then an error.

    Inputs: section (one phases[] entry), default_dojos (the run's
        non-held-out dojos; also what a table's name patterns match).
    Output: Phase.
    Side effects: none.
    Exceptions: ValueError as parse_run_config.
    """
    diet_rule = parse_diet(section.required_section("diet"), default_dojos)
    dojo_names = _phase_dojo_names(section, diet_rule, default_dojos)
    fields: dict[str, Any] = {
        "name": section.required("name", str),
        "dojo_names": dojo_names,
        "diet_rule": diet_rule,
        "encoder_trainable": section.required("encoder_trainable", bool),
        "encoder_lr": section.required("encoder_lr", float),
        "head_lr": section.required("head_lr", float),
        "steps_per_round": section.required("steps_per_round", int),
        "max_rounds": section.required("max_rounds", int),
        "saturation": build_flat_dataclass(
            SaturationSpec, section.required_section("saturation")
        ),
    }
    # A dict (not keywords) so an absent max_grad_norm or lr_schedule keeps
    # Phase's own default rather than a copy of it here
    if section.has("max_grad_norm"):
        fields["max_grad_norm"] = section.required("max_grad_norm", float)
    if section.has("lr_schedule"):
        fields["lr_schedule"] = build_flat_dataclass(
            LearningRateSchedule, section.required_section("lr_schedule")
        )
    section.reject_unread_keys()
    return construct_at(section.where, lambda: Phase(**fields))


def _phase_dojo_names(
    section: ConfigSection, diet_rule: DietRule, default_dojos: tuple[str, ...]
) -> tuple[str, ...]:
    """A phase's dojos: its table diet's, else its "dojos" key, else
    default_dojos.

    Inputs: section (the phases[] entry), diet_rule (already parsed),
        default_dojos.
    Output: tuple[str, ...].
    Side effects: marks "dojos" read.
    Exceptions: ValueError if a table-diet phase also lists "dojos".
    """
    if isinstance(diet_rule, TableDiet):
        if section.has("dojos"):
            raise ValueError(
                f"{section.location_of('dojos')}: a table diet names the "
                "phase's dojos itself; drop this list"
            )
        return diet_rule.dojo_names
    if section.has("dojos"):
        return section.required_str_list("dojos")
    return default_dojos


_EXCLUDE_PREFIX = "!"
_PATTERN_CHARACTERS = frozenset("*?[")


def _expanded_dojo_entries(
    entries: tuple[str, ...], catalog_names: Sequence[str], where: str
) -> tuple[str, ...]:
    """A dojos list with its patterns expanded, applied in order:
    a plain name is added unless already there (written twice it is kept
    twice, so RunConfig's duplicate check catches the copy-paste slip);
    a pattern ("*", "?", "[...]", case
    sensitive) adds every catalog name it matches, in catalog order, that
    is not already in the list; "!pattern" removes every name in the list
    so far that it matches.

    Inputs: entries, catalog_names, where (for errors).
    Output: tuple[str, ...].
    Side effects: none.
    Exceptions: ValueError naming an entry (pattern or exclusion) that
        matches nothing.

    Example:
        >>> _expanded_dojo_entries(
        ...     ("contrastive.*", "!*.gwent"), ("contrastive.mtg", "contrastive.gwent"), "dojos"
        ... )
        ('contrastive.mtg',)
    """
    result: list[str] = []
    written_names: set[str] = set()
    # Apply each entry in order, so an exclusion sees what came before it
    for index, entry in enumerate(entries):
        if entry.startswith(_EXCLUDE_PREFIX):
            pattern = entry.removeprefix(_EXCLUDE_PREFIX)
            kept = [name for name in result if not fnmatch.fnmatchcase(name, pattern)]
            if len(kept) == len(result):
                raise ValueError(f"{where}.{index}: {entry!r} excludes nothing")
            result = kept
        elif _PATTERN_CHARACTERS.isdisjoint(entry):
            if entry in written_names or entry not in result:
                result.append(entry)
            written_names.add(entry)
        else:
            matches = [
                name for name in catalog_names if fnmatch.fnmatchcase(name, entry)
            ]
            if not matches:
                raise ValueError(f"{where}.{index}: {entry!r} matches no dojo")
            result.extend(name for name in matches if name not in result)
    return tuple(result)


def _parse_holdout(section: ConfigSection) -> HoldoutSpec:
    """{"seed", "tier_ratios": [train, test, validation], "held_out_games":
    [GameId values], "deck_split_ratios": [train, test, validation]} ->
    HoldoutSpec; held_out_games optional (empty), deck_split_ratios
    optional (HoldoutSpec's default).

    Inputs: section ("holdout"). Output: HoldoutSpec.
    Side effects: none.
    Exceptions: ValueError for a ratio list that is not three numbers, or
        an unknown game.
    """
    seed = section.required("seed", int)
    tier_ratios = _parse_ratios(section, "tier_ratios")
    deck_split_ratios = (
        _parse_ratios(section, "deck_split_ratios")
        if section.has("deck_split_ratios")
        else DEFAULT_DECK_SPLIT_RATIOS
    )
    games = frozenset(
        _parse_game(value, section.location_of("held_out_games"))
        for value in section.optional_str_list("held_out_games")
    )
    section.reject_unread_keys()
    return construct_at(
        section.where,
        lambda: HoldoutSpec(
            seed=seed,
            tier_ratios=tier_ratios,
            held_out_games=games,
            deck_split_ratios=deck_split_ratios,
        ),
    )


def _parse_ratios(section: ConfigSection, key: str) -> tuple[float, float, float]:
    """section[key] as (train, test, validation) ratios.

    Inputs: section, key (a required three-number list).
    Output: three floats. Side effects: marks key read.
    Exceptions: ValueError naming the location if key is missing or not a
        list of three numbers.
    """
    location = section.location_of(key)
    ratios = section.required_list(key)
    if len(ratios) != _TIER_COUNT:
        raise ValueError(f"{location} must have {_TIER_COUNT} numbers")
    train, test, validation = (
        checked_scalar(ratio, float, f"{location}.{index}")
        for index, ratio in enumerate(ratios)
    )
    return (train, test, validation)


def _parse_game(value: str, where: str) -> GameId:
    """GameId(value), with a message naming the valid games.

    Inputs: value, where (for the message). Output: GameId.
    Side effects: none. Exceptions: ValueError for an unknown game.
    """
    try:
        return GameId(value)
    except ValueError as error:
        choices = sorted(game.value for game in GameId)
        raise ValueError(
            f"{where}: unknown game {value!r}, want one of {choices}"
        ) from error


def _parse_staple_thresholds(section: ConfigSection) -> dict[str, float]:
    """The `staple_subsampling:` section: contrastive dojo name -> t, the
    staple-subsampling threshold (each card kept with probability
    min(1, sqrt(t / df))). Leave a dojo out for t = inf (no subsampling).

        staple_subsampling:
          contrastive.dominion: 0.1

    Inputs: section ("staple_subsampling", possibly empty). Output: dict.
    Side effects: none.
    Exceptions: ValueError for a value that is not a finite number > 0.
    """
    result: dict[str, float] = {}
    for name in section.keys():
        threshold = section.required(name, float)
        if threshold <= 0:
            raise ValueError(f"{section.location_of(name)} must be > 0")
        result[name] = threshold
    section.reject_unread_keys()
    return result


def _parse_regression_loss(top: ConfigSection) -> RegressionLossKind:
    """The top-level `regression_loss:` key -> RegressionLossKind (default
    mse, today's behavior).

        regression_loss: huber

    Inputs: top (the config's top ConfigSection). Output: RegressionLossKind.
    Side effects: marks the key read.
    Exceptions: ValueError, naming the key's config location, for a value
        that is not one of the kind names.
    """
    return top.optional_choice(
        "regression_loss", RegressionLossKind, RegressionLossKind.MSE
    )


def _parse_loss_weighting(top: ConfigSection) -> LossWeighting | None:
    """The `loss_weights:` mapping and `baseline_floor:` scalar at the top
    level of the config -> LossWeighting. Both are optional: no weights and
    the default floor still normalize every dojo by its baseline.

        weight_by_baseline: false    # optional, default true

    `weight_by_baseline: false` returns None (train on the raw loss, the
    weights-off control for an experiment); it is an error to combine it
    with `loss_weights:` or `baseline_floor:`.

        loss_weights:
          cross_game.rarity_tier: 2.0
          mtg.card_cmc: 0.75
        baseline_floor: 0.05

    The weight keys are dojo names (dotted), so `--set` overrides cannot
    address them; edit the file.

    Inputs: top (the config's top ConfigSection).
    Output: LossWeighting, or None when weight_by_baseline is false.
    Side effects: marks the keys read.
    Exceptions: ValueError, naming the key's config location, for a weight
        or floor that is not a finite number > 0 (checked here, like
        _parse_staple_thresholds, before LossWeighting is built), or for
        weights or a floor given alongside weight_by_baseline: false.
    """
    # weight_by_baseline: false is the raw-loss control; nothing else may be set
    if not top.optional("weight_by_baseline", bool, True):
        if top.has("loss_weights") or top.has("baseline_floor"):
            raise ValueError(
                f"{top.location_of('weight_by_baseline')} is false, so "
                "loss_weights and baseline_floor have no effect; remove them"
            )
        return None

    weights: dict[str, float] = {}
    section = top.optional_section("loss_weights")
    for name in section.keys():
        weights[name] = _read_positive_finite(section, name)
    section.reject_unread_keys()
    floor = _read_positive_finite(top, "baseline_floor", DEFAULT_BASELINE_FLOOR)
    return LossWeighting(weights, floor)


def _read_positive_finite(
    section: ConfigSection, key: str, default: float | None = None
) -> float:
    """The float at key, which must be finite and > 0.

    Inputs: section, key, default (used when key is absent; None makes the
        key required). Output: float. Side effects: marks key read.
    Exceptions: ValueError naming the key's location if missing without a
        default, wrongly typed, or not finite and > 0.
    """
    if default is None:
        value = section.required(key, float)
    else:
        value = section.optional(key, float, default)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{section.location_of(key)} must be finite and > 0")
    return value


def _parse_device(text: str, where: str) -> torch.device:
    """torch.device(text), so a malformed string fails at parse time.

    Inputs: text (e.g. "cuda", "cuda:0", "cpu"), where (for the message).
    Output: torch.device.
    Side effects: none (availability is the driver's check).
    Exceptions: ValueError for a string torch rejects.
    """
    try:
        return torch.device(text)
    except RuntimeError as error:
        raise ValueError(f"{where} {text!r}: {error}") from error


def _split_override(override: str) -> tuple[list[str], Any]:
    """Split "a.0.b=value" into its path and YAML-parsed value.

    Inputs: override (str). Output: (["a", "0", "b"], value).
    Side effects: none.
    Exceptions: ValueError without "=" or with an empty path segment.
    """
    key, separator, raw_value = override.partition("=")
    if not separator:
        raise ValueError(f"override {override!r} must look like key.path=value")
    path = key.strip().split(".")
    if not all(path):
        raise ValueError(f"override {override!r} has an empty key segment")
    return path, yaml.safe_load(raw_value)


def _walk_to_parent(
    document: ConfigDocument, path: list[str]
) -> tuple[dict[str, Any] | list[Any], str]:
    """Follow every segment but the last; return (container, last segment).

    Iterative (a loop over segments), not recursive.
    Inputs: document, path (non-empty). Output: (container, last segment).
    Side effects: none.
    Exceptions: ValueError for a missing key, an out-of-range or
        non-integer list index, or a scalar in the way.
    """
    current: Any = document
    for depth, segment in enumerate(path[:-1]):
        where = ".".join(path[: depth + 1])
        if isinstance(current, dict):
            if segment not in current:
                raise ValueError(f"override path {where!r} does not exist")
            current = current[segment]
        elif isinstance(current, list):
            current = current[_list_index(segment, len(current), where)]
        else:
            raise ValueError(f"override path {where!r} goes through a scalar")
    if not isinstance(current, (dict, list)):
        raise ValueError(f"override path {'.'.join(path)!r} goes through a scalar")
    return current, path[-1]


def _set_in_container(
    container: dict[str, Any] | list[Any], key: str, value: Any, where: str
) -> None:
    """container[key] = value, key parsed as an index for a list.

    Inputs: container, key, value, where (the full override path, for the
        message). Output: none.
    Side effects: mutates container.
    Exceptions: ValueError for a non-integer or out-of-range list index.
    """
    if isinstance(container, list):
        container[_list_index(key, len(container), where)] = value
    else:
        container[key] = value


def _list_index(segment: str, length: int, where: str) -> int:
    """segment as an index into a list of that length.

    Inputs: segment, length, where (for the message). Output: int.
    Side effects: none.
    Exceptions: ValueError if segment is not a non-negative integer below
        length.
    """
    if not segment.isdigit() or int(segment) >= length:
        raise ValueError(f"override path {where!r}: no list index {segment!r}")
    return int(segment)
