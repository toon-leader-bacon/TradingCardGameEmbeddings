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
"""

import copy
import dataclasses
import fnmatch
import math
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Literal,
    Mapping,
    Sequence,
    TypeVar,
    cast,
    get_origin,
)

import torch
import yaml

from src.dojos.mods.card_field_mods import FieldMask
from src.dojos.mods.mod_specs import (
    CardDropoutSpec,
    CardSubsampleSpec,
    DuplicateCollapseSpec,
    ModSpec,
    RandomKeyMaskSpec,
    ShuffleKeysSpec,
    WeightedFieldMaskSpec,
)
from src.schema.game_id import GameId
from src.schema.holdout import HoldoutSpec
from src.dojos.loss.regression_objective import RegressionLossKind
from src.training.loss_weighting import DEFAULT_BASELINE_FLOOR, LossWeighting
from src.training.plan import (
    DietRule,
    DietTableRow,
    DojoGroupRow,
    DojoRow,
    FaultPolicy,
    FlatDietRule,
    HardwareLimits,
    Phase,
    Proportional,
    SaturationSpec,
    SubTableRow,
    TableDiet,
    Temperature,
    TrainingPlan,
    Uniform,
)
from src.utils.drop_table import DropTable

# The raw, untyped YAML mapping. Exists only between the file and
# parse_run_config (and in the run directory's written copy).
ConfigDocument = dict[str, Any]

_T = TypeVar("_T")

# Diet rules that take no parameters, by their config name
_PARAMETERLESS_DIETS: dict[str, Callable[[], FlatDietRule]] = {
    "uniform": Uniform,
    "proportional": Proportional,
}
_TEMPERATURE_DIET = "temperature"
_TABLE_DIET = "table"
_TIER_COUNT = 3


class EmbeddingHeadKind(StrEnum):
    """Which EmbeddingHead (src/encoder_model/embedding_head.py) turns the
    text encoder's tokens into one card embedding."""

    LINEAR_PROJECTION = "linear_projection"
    RESIDUAL_MLP = "residual_mlp"
    ATTENTION_POOLING = "attention_pooling"


@dataclass(frozen=True)
class GroupAttentionSpec:
    """The self-attention layers a multi-card model stacks on the
    embedding head (MultiCardModel), so cards in a group see each other.

    num_layers, num_heads: Transformer encoder layers and attention heads
        per layer (num_heads must divide card_embedding_size).
    ffn_dim: each layer's feed-forward width.
    dropout: dropout inside each layer.
    norm_first: pre-norm (LayerNorm before attention and feed-forward)
        rather than torch's default post-norm.
    """

    num_layers: int = 2
    num_heads: int = 4
    ffn_dim: int = 1024
    dropout: float = 0.1
    norm_first: bool = True

    def __post_init__(self) -> None:
        for name in ("num_layers", "num_heads", "ffn_dim"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        if not 0 <= self.dropout < 1:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")


@dataclass(frozen=True)
class ModelSpec:
    """The encoder to build: a frozen pretrained text encoder, an embedding
    head, and for a multi-card model, group attention on top.

    embedding_head: which EmbeddingHead.
    checkpoint: Hugging Face checkpoint of the pretrained text encoder.
    card_embedding_size: output embedding width; every dojo head is built
        for it.
    mlp_hidden_dim, mlp_num_blocks: the residual-MLP head's width and
        block count (read only for EmbeddingHeadKind.RESIDUAL_MLP).
    group_attention: None for a single-card model (each card embedded
        alone); otherwise the MultiCardModel's self-attention layers.
    """

    embedding_head: EmbeddingHeadKind
    checkpoint: str
    card_embedding_size: int
    mlp_hidden_dim: int = 512
    mlp_num_blocks: int = 3
    group_attention: GroupAttentionSpec | None = None

    def __post_init__(self) -> None:
        if self.card_embedding_size < 1:
            raise ValueError("card_embedding_size must be >= 1")
        if self.mlp_hidden_dim < 1 or self.mlp_num_blocks < 0:
            raise ValueError("mlp_hidden_dim must be >= 1, mlp_num_blocks >= 0")
        attention = self.group_attention
        if attention is not None and self.card_embedding_size % attention.num_heads:
            raise ValueError(
                f"group_attention.num_heads ({attention.num_heads}) must divide "
                f"card_embedding_size ({self.card_embedding_size})"
            )


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
    entries, see _parse_mod_spec), staple_subsampling (optional: dojo name
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
    faults = _build_flat_dataclass(FaultPolicy, top.optional_section("faults"))
    loss_weighting = _parse_loss_weighting(top)
    plan = _construct_at(
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
    model = _parse_model(top.required_section("model"))
    limits = _build_flat_dataclass(HardwareLimits, top.required_section("hardware"))
    regression_loss = _parse_regression_loss(top)
    mod_overrides = _parse_mod_overrides(top.optional_section("mods"))
    staple_thresholds = _parse_staple_thresholds(
        top.optional_section("staple_subsampling")
    )
    top.reject_unread_keys()
    result = _construct_at(
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
        ),
    )
    return result


class ConfigSection:
    """Typed, key-checked reading of one mapping in the raw document.

    Every getter records the key it read, so reject_unread_keys() can flag
    typos; nested sections come back as ConfigSections carrying their
    dotted location, so errors name where they are. Each parser calls
    reject_unread_keys() on the sections it reads. bool is never accepted
    where int or float is expected (bool subclasses int in Python); an int
    is widened where float is expected, and so is a numeric string,
    because PyYAML reads "1e-3" (no decimal point) as a string.
    """

    def __init__(self, raw: Any, where: str) -> None:
        """Inputs: raw (should be a mapping), where (its dotted location,
        e.g. "phases.0"). Output: none. Side effects: none.
        Exceptions: ValueError if raw is not a mapping with str keys."""
        if not isinstance(raw, dict) or not all(isinstance(key, str) for key in raw):
            raise ValueError(f"{where} must be a mapping with string keys")
        self._raw: dict[str, Any] = raw
        self._where = where
        self._read_keys: set[str] = set()

    @property
    def where(self) -> str:
        """This section's dotted location. Side effects: none.
        Exceptions: none."""
        return self._where

    def location_of(self, key: str) -> str:
        """The dotted location of key, for error messages.
        Inputs: key. Output: str. Side effects: none. Exceptions: none."""
        return f"{self._where}.{key}"

    def required(self, key: str, expected_type: type[_T]) -> _T:
        """The scalar at key, checked against expected_type.

        Inputs: key, expected_type (int, float, str or bool).
        Output: the value (an int or numeric string widened to float for
            float).
        Side effects: records key as read.
        Exceptions: ValueError if missing or wrongly typed.
        """
        return _checked_scalar(self._take(key), expected_type, self.location_of(key))

    def optional(self, key: str, expected_type: type[_T], default: _T) -> _T:
        """As required(), but default when key is absent.

        Inputs: key, expected_type, default. Output: the value or default.
        Side effects: records key as read.
        Exceptions: ValueError if present and wrongly typed.
        """
        if key not in self._raw:
            return default
        return self.required(key, expected_type)

    def required_str_list(self, key: str) -> tuple[str, ...]:
        """The list of strings at key.

        Inputs: key. Output: tuple[str, ...].
        Side effects: records key as read.
        Exceptions: ValueError if missing, not a list, or any item is not
            a str.
        """
        items = self._take_list(key)
        return tuple(
            _checked_scalar(item, str, f"{self.location_of(key)}.{index}")
            for index, item in enumerate(items)
        )

    def optional_str_list(self, key: str) -> tuple[str, ...]:
        """As required_str_list(), but () when key is absent.

        Inputs: key. Output: tuple[str, ...].
        Side effects: records key as read.
        Exceptions: ValueError if present and malformed.
        """
        if key not in self._raw:
            return ()
        return self.required_str_list(key)

    def required_list(self, key: str) -> list[Any]:
        """The raw list at key; its items are the caller's to check.

        Inputs: key. Output: list[Any].
        Side effects: records key as read.
        Exceptions: ValueError if missing or not a list.
        """
        return self._take_list(key)

    def required_section(self, key: str) -> "ConfigSection":
        """The nested mapping at key, located at "<where>.<key>".

        Inputs: key. Output: ConfigSection.
        Side effects: records key as read.
        Exceptions: ValueError if missing or not a mapping.
        """
        return ConfigSection(self._take(key), self.location_of(key))

    def optional_section(self, key: str) -> "ConfigSection":
        """As required_section(), but an empty section when key is absent.

        Inputs: key. Output: ConfigSection.
        Side effects: records key as read.
        Exceptions: ValueError if present and not a mapping.
        """
        if key not in self._raw:
            return ConfigSection({}, self.location_of(key))
        return self.required_section(key)

    def required_section_list(self, key: str) -> list["ConfigSection"]:
        """The list of mappings at key, each located at "<where>.<key>.<i>".

        Inputs: key. Output: list[ConfigSection].
        Side effects: records key as read.
        Exceptions: ValueError if missing, not a list, or any item is not
            a mapping.
        """
        items = self._take_list(key)
        return [
            ConfigSection(item, f"{self.location_of(key)}.{index}")
            for index, item in enumerate(items)
        ]

    def keys(self) -> list[str]:
        """Every key present, in file order (for sections whose keys are
        data, like `mods:`). Does not count them as read.
        Inputs: none. Output: list[str]. Side effects: none.
        Exceptions: none."""
        return list(self._raw)

    def has(self, key: str) -> bool:
        """Whether key is present. Inputs: key. Output: bool.
        Side effects: none (does not count as reading). Exceptions: none."""
        return key in self._raw

    def reject_unread_keys(self) -> None:
        """Inputs: none. Output: none. Side effects: none.
        Exceptions: ValueError naming (with location) every key never read."""
        unread = sorted(set(self._raw) - self._read_keys)
        if unread:
            locations = [self.location_of(key) for key in unread]
            raise ValueError(f"unknown config keys {locations}")

    def _take(self, key: str) -> Any:
        """The raw value at key. Side effects: records key as read.
        Exceptions: ValueError if missing."""
        if key not in self._raw:
            raise ValueError(f"{self.location_of(key)} is required")
        self._read_keys.add(key)
        return self._raw[key]

    def _take_list(self, key: str) -> list[Any]:
        """The raw list at key. Side effects: records key as read.
        Exceptions: ValueError if missing or not a list."""
        value = self._take(key)
        if not isinstance(value, list):
            raise ValueError(f"{self.location_of(key)} must be a list")
        return value


def _parse_phase(section: ConfigSection, default_dojos: tuple[str, ...]) -> Phase:
    """One phases[] entry -> Phase; "dojos" defaults to default_dojos. A
    table diet names the phase's dojos itself, so "dojos" is then an error.

    Inputs: section (one phases[] entry), default_dojos (the run's
        non-held-out dojos; also what a table's name patterns match).
    Output: Phase.
    Side effects: none.
    Exceptions: ValueError as parse_run_config.
    """
    diet_rule = _parse_diet(section.required_section("diet"), default_dojos)
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
        "saturation": _build_flat_dataclass(
            SaturationSpec, section.required_section("saturation")
        ),
    }
    # A dict (not keywords) so an absent max_grad_norm keeps Phase's own
    # default rather than a copy of it here
    if section.has("max_grad_norm"):
        fields["max_grad_norm"] = section.required("max_grad_norm", float)
    section.reject_unread_keys()
    return _construct_at(section.where, lambda: Phase(**fields))


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


def _parse_diet(section: ConfigSection, run_dojos: tuple[str, ...]) -> DietRule:
    """A phase's "diet" -> DietRule: {"rule": "table", "table": [<rows>]}
    (see _parse_table_rows), or a flat rule (see _parse_flat_diet).

    Inputs: section (a phase's "diet"), run_dojos (the run's non-held-out
        dojos, which a table's name patterns match).
    Output: DietRule.
    Side effects: none.
    Exceptions: ValueError as _parse_flat_diet and _parse_table_rows.

    Example:
        >>> _parse_diet(ConfigSection({"rule": "uniform"}, "diet"), ("a",))
        Uniform()
    """
    if section.has("rule") and section.required("rule", str) == _TABLE_DIET:
        rows = _parse_table_rows(section.required_section_list("table"), run_dojos)
        section.reject_unread_keys()
        return _construct_at(section.where, lambda: TableDiet(rows))
    return _parse_flat_diet(section, other_choices=(_TABLE_DIET,))


def _parse_table_rows(
    rows: list[ConfigSection], run_dojos: tuple[str, ...]
) -> tuple[DietTableRow, ...]:
    """A diet `table:` list -> its rows. Each row has a weight and exactly
    one of:

        {weight: 1, dojo: contrastive.mtg}             DojoRow
        {weight: 1, dojos: ["gwent_one.*"],            DojoGroupRow; within
         within: {rule: temperature, alpha: 0.3}}      optional (uniform)
        {weight: 50, table: [<rows>]}                  SubTableRow

    Recursive over nested tables, mirroring the YAML's own shape (as
    _parse_mask_table).

    Inputs: rows (each a ConfigSection located at its index), run_dojos.
    Output: tuple[DietTableRow, ...].
    Side effects: none.
    Exceptions: ValueError for a row with none or several of dojo/dojos/
        table, a bad weight, a pattern matching no run dojo, or a bad
        within rule.
    """
    result: list[DietTableRow] = []

    # Each row is one dojo, a matched group, or a sub-table
    for row in rows:
        weight = row.required("weight", float)
        kind = _table_row_kind(row)
        if kind == "dojo":
            entry: DietTableRow = _construct_at(
                row.where, lambda: DojoRow(weight, row.required("dojo", str))
            )
        elif kind == "dojos":
            entry = _parse_dojo_group_row(row, weight, run_dojos)
        else:
            sub_rows = _parse_table_rows(row.required_section_list("table"), run_dojos)
            entry = _construct_at(row.where, lambda: SubTableRow(weight, sub_rows))
        row.reject_unread_keys()
        result.append(entry)
    return tuple(result)


def _table_row_kind(row: ConfigSection) -> Literal["dojo", "dojos", "table"]:
    """Which one of dojo/dojos/table a diet table row has.

    Inputs: row. Output: the key present.
    Side effects: none.
    Exceptions: ValueError if none or more than one is present.
    """
    kinds: tuple[Literal["dojo", "dojos", "table"], ...] = ("dojo", "dojos", "table")
    present = [kind for kind in kinds if row.has(kind)]
    if len(present) != 1:
        raise ValueError(
            f"{row.where} needs exactly one of dojo, dojos or table; has {present}"
        )
    return present[0]


def _parse_dojo_group_row(
    row: ConfigSection, weight: float, run_dojos: tuple[str, ...]
) -> DojoGroupRow:
    """A `dojos:` row -> DojoGroupRow: its patterns matched against
    run_dojos (_matching_dojos), its optional "within" a flat rule
    (default uniform).

    Inputs: row, weight (already read), run_dojos.
    Output: DojoGroupRow.
    Side effects: marks "dojos" and "within" read.
    Exceptions: ValueError as _matching_dojos and _parse_flat_diet.
    """
    dojo_names = _matching_dojos(
        row.required_str_list("dojos"), run_dojos, row.location_of("dojos")
    )
    within: FlatDietRule = Uniform()
    if row.has("within"):
        within = _parse_flat_diet(row.required_section("within"))
    return _construct_at(row.where, lambda: DojoGroupRow(weight, dojo_names, within))


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


def _matching_dojos(
    patterns: tuple[str, ...], run_dojos: tuple[str, ...], where: str
) -> tuple[str, ...]:
    """Every run dojo matching any pattern (fnmatch-style, case sensitive:
    "*", "?", "[...]"), in run_dojos order, each once.

    Inputs: patterns, run_dojos, where (for errors).
    Output: tuple[str, ...], non-empty.
    Side effects: none.
    Exceptions: ValueError naming any pattern that matches no run dojo
        (a typo, or a held-out dojo).
    """
    unmatched = [
        pattern
        for pattern in patterns
        if not any(fnmatch.fnmatchcase(name, pattern) for name in run_dojos)
    ]
    if unmatched:
        raise ValueError(
            f"{where}: {unmatched} match no dojo in dojos (held-out dojos "
            "are not matched)"
        )
    return tuple(
        name
        for name in run_dojos
        if any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)
    )


def _parse_flat_diet(
    section: ConfigSection, other_choices: tuple[str, ...] = ()
) -> FlatDietRule:
    """{"rule": "uniform" | "proportional" | "temperature", "alpha": ...}
    -> the matching rule; alpha only (and required) for temperature.

    Inputs: section (a phase's "diet", or a table row's "within"),
        other_choices (rules valid where section sits that are not flat,
        named in the unknown-rule error).
    Output: FlatDietRule.
    Side effects: none.
    Exceptions: ValueError for an unknown rule or a misplaced/missing alpha.
    """
    rule = section.required("rule", str)
    result: FlatDietRule
    if rule == _TEMPERATURE_DIET:
        alpha = section.required("alpha", float)
        result = _construct_at(section.where, lambda: Temperature(alpha=alpha))
    elif rule in _PARAMETERLESS_DIETS:
        result = _PARAMETERLESS_DIETS[rule]()
    else:
        choices = sorted([*_PARAMETERLESS_DIETS, _TEMPERATURE_DIET, *other_choices])
        raise ValueError(
            f"{section.location_of('rule')} is {rule!r}, want one of {choices}"
        )
    # A misplaced alpha (on uniform/proportional) is an unread key
    section.reject_unread_keys()
    return result


def _parse_holdout(section: ConfigSection) -> HoldoutSpec:
    """{"seed", "tier_ratios": [train, test, validation], "held_out_games":
    [GameId values]} -> HoldoutSpec; held_out_games optional (empty).

    Inputs: section ("holdout"). Output: HoldoutSpec.
    Side effects: none.
    Exceptions: ValueError for a ratio list that is not three numbers, or
        an unknown game.
    """
    seed = section.required("seed", int)
    ratios_location = section.location_of("tier_ratios")
    ratios = section.required_list("tier_ratios")
    if len(ratios) != _TIER_COUNT:
        raise ValueError(f"{ratios_location} must have {_TIER_COUNT} numbers")
    train, test, validation = (
        _checked_scalar(ratio, float, f"{ratios_location}.{index}")
        for index, ratio in enumerate(ratios)
    )
    games = frozenset(
        _parse_game(value, section.location_of("held_out_games"))
        for value in section.optional_str_list("held_out_games")
    )
    section.reject_unread_keys()
    return _construct_at(
        section.where,
        lambda: HoldoutSpec(
            seed=seed, tier_ratios=(train, test, validation), held_out_games=games
        ),
    )


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


def _parse_model(section: ConfigSection) -> ModelSpec:
    """{"embedding_head", "checkpoint", "card_embedding_size",
    ["mlp_hidden_dim", "mlp_num_blocks"], ["group_attention"]} -> ModelSpec.

    Inputs: section ("model"). Output: ModelSpec.
    Side effects: none.
    Exceptions: ValueError for an unknown embedding_head, mlp_* keys on a
        head other than residual_mlp, or an invalid size.
    """
    # Which head; the mlp_* sizes only mean something for the residual MLP
    head_text = section.required("embedding_head", str)
    if head_text not in {kind.value for kind in EmbeddingHeadKind}:
        choices = [kind.value for kind in EmbeddingHeadKind]
        raise ValueError(
            f"{section.location_of('embedding_head')} is {head_text!r}, "
            f"want one of {choices}"
        )
    head = EmbeddingHeadKind(head_text)
    mlp_keys = [key for key in ("mlp_hidden_dim", "mlp_num_blocks") if section.has(key)]
    if mlp_keys and head is not EmbeddingHeadKind.RESIDUAL_MLP:
        raise ValueError(
            f"{section.where}: {mlp_keys} apply only to embedding_head: residual_mlp"
        )

    # Sizes, then group attention (absent = a single-card model)
    checkpoint = section.required("checkpoint", str)
    card_embedding_size = section.required("card_embedding_size", int)
    mlp_hidden_dim = section.optional("mlp_hidden_dim", int, ModelSpec.mlp_hidden_dim)
    mlp_num_blocks = section.optional("mlp_num_blocks", int, ModelSpec.mlp_num_blocks)
    group_attention = None
    if section.has("group_attention"):
        group_attention = _parse_group_attention(
            section.required_section("group_attention")
        )
    section.reject_unread_keys()
    return _construct_at(
        section.where,
        lambda: ModelSpec(
            head,
            checkpoint,
            card_embedding_size,
            mlp_hidden_dim,
            mlp_num_blocks,
            group_attention,
        ),
    )


def _parse_group_attention(section: ConfigSection) -> GroupAttentionSpec:
    """`model.group_attention` -> GroupAttentionSpec; an absent key keeps
    the dataclass default.

    Private helper - single caller is _parse_model().
    Inputs: section. Output: GroupAttentionSpec. Side effects: none.
    Exceptions: ValueError for a wrongly typed, unknown or invalid value.
    """
    defaults = GroupAttentionSpec()
    num_layers = section.optional("num_layers", int, defaults.num_layers)
    num_heads = section.optional("num_heads", int, defaults.num_heads)
    ffn_dim = section.optional("ffn_dim", int, defaults.ffn_dim)
    dropout = section.optional("dropout", float, defaults.dropout)
    norm_first = section.optional("norm_first", bool, defaults.norm_first)
    section.reject_unread_keys()
    return _construct_at(
        section.where,
        lambda: GroupAttentionSpec(num_layers, num_heads, ffn_dim, dropout, norm_first),
    )


def _parse_mod_overrides(section: ConfigSection) -> dict[str, tuple[ModSpec, ...]]:
    """The `mods:` section: dojo name -> its augmentation specs.

    RunConfig checks each name is a run dojo; build_dojos checks any deck
    spec against DECK_MOD_GROUPS. The list replaces the dojo's game
    defaults; an empty list means "no augmentation" for that dojo.
    (A dojo name contains a ".", so --set cannot address it; edit the
    file instead.)

    Inputs: section ("mods", possibly empty). Output: dict.
    Side effects: none.
    Exceptions: ValueError for any malformed mod entry.
    """
    result: dict[str, tuple[ModSpec, ...]] = {}

    # Parse each dojo's list of mod entries
    for name in section.keys():
        result[name] = tuple(
            _parse_mod_spec(entry) for entry in section.required_section_list(name)
        )
    section.reject_unread_keys()
    return result


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
    text = top.optional("regression_loss", str, RegressionLossKind.MSE.value)
    try:
        return RegressionLossKind(text)
    except ValueError:
        wanted = [kind.value for kind in RegressionLossKind]
        raise ValueError(
            f"{top.location_of('regression_loss')} is {text!r}, want one of {wanted}"
        ) from None


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


def _parse_mod_spec(section: ConfigSection) -> ModSpec:
    """One mod entry -> ModSpec, by its "kind":

        {kind: shuffle_keys}
        {kind: random_key_mask, probability: 0.1}
        {kind: weighted_field_mask, table: [<rows>]}   (see _parse_mask_table)
        {kind: card_dropout, drop_probability: 0.1, groups: [0]}
        {kind: card_subsample, keep_fraction: 0.8, groups: [1]}
        {kind: duplicate_collapse}

    The last three are deck mods (groups optional, default [0]); only the
    dojos in dojo_catalog.DECK_MOD_GROUPS take them, for the groups listed
    there.

    Dispatch is a kind -> parser table, not an if-chain.

    Inputs: section (one entry). Output: ModSpec.
    Side effects: none.
    Exceptions: ValueError for an unknown kind, or a malformed entry
        (including unknown keys).
    """
    kind = section.required("kind", str)
    parser = _MOD_PARSERS.get(kind)
    if parser is None:
        raise ValueError(
            f"{section.location_of('kind')} is {kind!r}, "
            f"want one of {sorted(_MOD_PARSERS)}"
        )
    result = _construct_at(section.where, lambda: parser(section))
    section.reject_unread_keys()
    return result


def _parse_mask_table(rows: list[ConfigSection], where: str) -> DropTable[FieldMask]:
    """A `table:` list -> DropTable of FieldMask. Each row has a weight and
    at most one of:

        {weight: 40}                                  mask nothing
        {weight: 35, mask: [[faction], [faction-duo]]}  FieldMask of paths
        {weight: 10, table: [<rows>]}                  roll on a sub-table

    A path is a list of steps (str keys, int list indexes). Recursive over
    nested tables, mirroring the YAML's own shape.

    Inputs: rows (each a ConfigSection located at its index), where (the
        table's own location, for an empty or all-zero table).
    Output: DropTable[FieldMask].
    Side effects: none.
    Exceptions: ValueError for a row with both mask and table, a bad
        weight, a malformed path, overlapping paths, or an empty table.
    """
    entries: list[tuple[float, FieldMask | DropTable[FieldMask]]] = []

    # Each row is "nothing", a mask, or a sub-table to roll on
    for row in rows:
        weight = row.required("weight", float)
        if row.has("mask") and row.has("table"):
            raise ValueError(f"{row.where} has both mask and table; pick one")
        outcome: FieldMask | DropTable[FieldMask] = FieldMask()
        if row.has("table"):
            outcome = _parse_mask_table(
                row.required_section_list("table"), row.location_of("table")
            )
        elif row.has("mask"):
            outcome = _parse_field_mask(
                row.required_list("mask"), row.location_of("mask")
            )
        row.reject_unread_keys()
        entries.append((weight, outcome))
    return _construct_at(where, lambda: DropTable.of(entries))


def _parse_field_mask(raw_paths: list[Any], where: str) -> FieldMask:
    """[[step, ...], ...] -> FieldMask (each inner list a FieldPath).

    Inputs: raw_paths, where. Output: FieldMask.
    Side effects: none.
    Exceptions: ValueError for a non-list path, a step that is neither
        str nor int, or what FieldMask's own validation rejects.
    """
    paths: list[tuple[str | int, ...]] = []
    for index, raw_path in enumerate(raw_paths):
        location = f"{where}.{index}"
        if not isinstance(raw_path, list) or not raw_path:
            raise ValueError(f"{location} must be a non-empty list of steps")
        for step in raw_path:
            # bool subclasses int; neither it nor a float is a step
            if isinstance(step, bool) or not isinstance(step, (str, int)):
                raise ValueError(f"{location}: step {step!r} must be a str or int")
        paths.append(tuple(raw_path))
    return _construct_at(where, lambda: FieldMask(tuple(paths)))


# Each mod kind's parser, by its config name
_MOD_PARSERS: dict[str, Callable[[ConfigSection], ModSpec]] = {
    "shuffle_keys": lambda section: ShuffleKeysSpec(),
    "random_key_mask": lambda section: RandomKeyMaskSpec(
        section.required("probability", float)
    ),
    "weighted_field_mask": lambda section: WeightedFieldMaskSpec(
        _parse_mask_table(
            section.required_section_list("table"), section.location_of("table")
        )
    ),
    "card_dropout": lambda section: CardDropoutSpec(
        section.required("drop_probability", float), _parse_groups(section)
    ),
    "card_subsample": lambda section: CardSubsampleSpec(
        section.required("keep_fraction", float), _parse_groups(section)
    ),
    "duplicate_collapse": lambda section: DuplicateCollapseSpec(_parse_groups(section)),
}


def _parse_groups(section: ConfigSection) -> tuple[int, ...]:
    """A deck mod entry's optional `groups:` list of group indexes; (0,),
    the deck of a multi-card input, when absent.

    Inputs: section (one mod entry). Output: tuple[int, ...].
    Side effects: records groups as read.
    Exceptions: ValueError if groups is not a list of ints.
    """
    if not section.has("groups"):
        return (0,)
    where = section.location_of("groups")
    return tuple(
        _checked_scalar(item, int, f"{where}.{index}")
        for index, item in enumerate(section.required_list("groups"))
    )


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


def _build_flat_dataclass(cls: type[_T], section: ConfigSection) -> _T:
    """Construct a frozen dataclass whose fields are all scalars
    (SaturationSpec, FaultPolicy, HardwareLimits) from a flat section.

    Reads each field of cls that the section has, typed by the field's
    annotation (int, float, str; a Literal alias such as Precision is read
    as str and left to cls's own validation); absent fields take cls's
    defaults. Unknown keys raise.

    Inputs: cls (a dataclass type), section.
    Output: an instance of cls.
    Side effects: none.
    Exceptions: ValueError for an unknown or missing field, a wrongly typed
        value, or whatever cls's own validation raises.
    """
    values: dict[str, Any] = {}
    for dataclass_field in dataclasses.fields(cast(Any, cls)):
        has_default = (
            dataclass_field.default is not dataclasses.MISSING
            or dataclass_field.default_factory is not dataclasses.MISSING
        )
        # Absent with a default: leave it to cls
        if not section.has(dataclass_field.name) and has_default:
            continue
        values[dataclass_field.name] = section.required(
            dataclass_field.name, _scalar_type_of(dataclass_field)
        )
    section.reject_unread_keys()
    return _construct_at(section.where, lambda: cls(**values))


def _scalar_type_of(field: "dataclasses.Field[Any]") -> type:
    """The scalar type a flat dataclass field is read as: its annotation,
    or str for a Literal.

    Inputs: field. Output: int, float, str or bool.
    Side effects: none.
    Exceptions: TypeError for any other annotation (a programming error:
        the dataclass is not flat).
    """
    if get_origin(field.type) is Literal:
        return str
    if field.type in (int, float, str, bool):
        return cast(type, field.type)
    raise TypeError(f"field {field.name!r} is not a scalar: {field.type!r}")


def _checked_scalar(value: Any, expected_type: type[_T], where: str) -> _T:
    """value, checked against expected_type (see ConfigSection's rules).

    Inputs: value, expected_type, where (for the message).
    Output: value as expected_type.
    Side effects: none.
    Exceptions: ValueError if value does not fit expected_type.
    """
    # bool subclasses int, so a stray `true` would pass as 1
    if isinstance(value, bool) and expected_type is not bool:
        raise ValueError(f"{where} must be {expected_type.__name__}, got a bool")
    if expected_type is float and isinstance(value, (int, float, str)):
        try:
            number = float(value)
        except (ValueError, OverflowError) as error:
            raise ValueError(f"{where} must be a number, got {value!r}") from error
        if not math.isfinite(number):
            raise ValueError(f"{where} must be finite, got {value!r}")
        return cast(_T, number)
    if not isinstance(value, expected_type):
        raise ValueError(
            f"{where} must be {expected_type.__name__}, got {type(value).__name__}"
        )
    return value


def _construct_at(where: str, build: Callable[[], _T]) -> _T:
    """build(), with any ValueError it raises prefixed by where.

    Inputs: where (dotted location), build (a constructor call).
    Output: build's result.
    Side effects: whatever build does (none for the frozen types).
    Exceptions: ValueError("<where>: <message>").
    """
    try:
        return build()
    except ValueError as error:
        raise ValueError(f"{where}: {error}") from error


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
