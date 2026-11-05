"""A run config's `mods:` section -> per-dojo augmentation specs
(src/dojos/mods/mod_specs.py)."""

from enum import StrEnum
from pathlib import Path
from typing import Any, Callable, TypeVar

from src.dojos.mods.card_field_mods import UNBOUNDED_VARIANTS, FieldMask
from src.dojos.mods.content_style_mods import BoolSpelling, KeyCase, ValueCase
from src.dojos.mods.mod_specs import (
    AugmentationRates,
    BoolToStringSpec,
    CardFieldModSpec,
    CardDropoutSpec,
    CardSubsampleSpec,
    DuplicateCollapseSpec,
    FlattenKeysSpec,
    KeyCaseSpec,
    ListToStringSpec,
    ModSpec,
    NumberToStringSpec,
    RandomKeyMaskSpec,
    ShuffleKeysSpec,
    StyleChoiceSpec,
    ValueCaseSpec,
    WeightedFieldMaskSpec,
)
from src.training.config_parsing.config_section import (
    ConfigSection,
    checked_scalar,
    construct_at,
    parse_by_kind,
)
from src.training.binder_mods_spec import BinderModsSpec
from src.utils.drop_table import DropTable

_StyleT = TypeVar("_StyleT", bound=StrEnum)


def parse_mod_overrides(section: ConfigSection) -> dict[str, tuple[ModSpec, ...]]:
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


def parse_binder_mods(section: ConfigSection) -> BinderModsSpec:
    """The `binder_mods:` section -> BinderModsSpec (see
    src/training/binder_mods.py):

        binder_mods:
          pipeline: [{kind: key_case, styles: [camel]}]   (required, non-empty)
          seed: 0                                          (optional, default 0)
          save_directory: data/cache/binder_mods/camel     (optional; "" = don't save)

    pipeline entries are the same as a dojo's `mods:` entries, card-field
    kinds only. default_augmentation never applies to them.

    Inputs: section ("binder_mods", present). Output: BinderModsSpec.
    Side effects: records the keys as read.
    Exceptions: ValueError (naming the location) for a missing or empty
        pipeline, a deck kind, any malformed entry, or an unknown key.

    Example:
        >>> parse_binder_mods(top.required_section("binder_mods"))
        BinderModsSpec(pipeline=(KeyCaseSpec(...),), seed=0, save_directory=None)
    """
    # Parse each pipeline entry, card-field kinds only
    # (BinderModsSpec rejects an empty pipeline; construct_at names where)
    pipeline = tuple(
        _card_field_spec(entry) for entry in section.required_section_list("pipeline")
    )

    # The scalars; "" means the same as no save_directory
    seed = section.optional("seed", int, 0)
    save_text = section.optional("save_directory", str, "")
    save_directory = Path(save_text) if save_text else None
    section.reject_unread_keys()
    return construct_at(
        section.where, lambda: BinderModsSpec(pipeline, seed, save_directory)
    )


def _card_field_spec(section: ConfigSection) -> CardFieldModSpec:
    """One binder pipeline entry -> CardFieldModSpec, by its "kind".

    Private helper - single consumer is parse_binder_mods.
    Inputs: section (one pipeline entry). Output: CardFieldModSpec.
    Side effects: records the entry's keys as read.
    Exceptions: ValueError naming section's location for a deck kind, or
        as _parse_mod_spec.
    """
    spec = _parse_mod_spec(section)
    if not isinstance(spec, CardFieldModSpec):
        raise ValueError(
            f"{section.where} is a deck mod; binder_mods takes card-field " "mods only"
        )
    return spec


def _parse_mod_spec(section: ConfigSection) -> ModSpec:
    """One mod entry -> ModSpec, by its "kind".

    Card-field kinds (src/dojos/mods/card_field_mods.py and
    content_style_mods.py):
        {kind: shuffle_keys}
        {kind: random_key_mask, probability: 0.1}      (probability required)
        {kind: weighted_field_mask, table: [<rows>]}   (see _parse_mask_table)
        {kind: key_case, styles: [camel, kebab]}
        {kind: flatten_keys, styles: [".", "/"]}
        {kind: value_case, styles: [lower]}
        {kind: bool_to_string, styles: [yes_no]}
        {kind: list_to_string, styles: [", "]}
        {kind: number_to_string}
    Every card-field kind takes an optional `variants:` int (default -1:
    negative draws fresh every time, 0 turns the mod off, N > 0 (at most
    100) gives each card at most N cacheable texts) and, apart from
    random_key_mask, an optional `probability:` (default 1.0), the chance
    the mod acts on a card at all. `styles:` (enum values, or separator
    strings) is optional; absent means the mod's DEFAULT_STYLES.

    Deck kinds (groups optional, default [0]); only the dojos in
    dojo_catalog.DECK_MOD_GROUPS take them, for the groups listed there:
        {kind: card_dropout, drop_probability: 0.1, groups: [0]}
        {kind: card_subsample, keep_fraction: 0.8, groups: [1]}
        {kind: duplicate_collapse}

    Dispatch is a kind -> parser table (parse_by_kind), not an if-chain.

    Inputs: section (one entry). Output: ModSpec.
    Side effects: none.
    Exceptions: ValueError for an unknown kind, or a malformed entry
        (including unknown keys).
    """
    return parse_by_kind(section, _MOD_PARSERS)


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
    return construct_at(where, lambda: DropTable.of(entries))


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
    return construct_at(where, lambda: FieldMask(tuple(paths)))


# Each mod kind's parser, by its config name
_MOD_PARSERS: dict[str, Callable[[ConfigSection], ModSpec]] = {
    "shuffle_keys": lambda section: ShuffleKeysSpec(
        variants=_parse_variants(section),
        probability=section.optional("probability", float, 1.0),
    ),
    "random_key_mask": lambda section: RandomKeyMaskSpec(
        section.required("probability", float), _parse_variants(section)
    ),
    "weighted_field_mask": lambda section: WeightedFieldMaskSpec(
        _parse_mask_table(
            section.required_section_list("table"), section.location_of("table")
        ),
        variants=_parse_variants(section),
        probability=section.optional("probability", float, 1.0),
    ),
    "key_case": lambda section: _style_spec(
        KeyCaseSpec, _parse_style_choices(section, KeyCase), section
    ),
    "flatten_keys": lambda section: _style_spec(
        FlattenKeysSpec, _parse_separators(section), section
    ),
    "value_case": lambda section: _style_spec(
        ValueCaseSpec, _parse_style_choices(section, ValueCase), section
    ),
    "bool_to_string": lambda section: _style_spec(
        BoolToStringSpec, _parse_style_choices(section, BoolSpelling), section
    ),
    "list_to_string": lambda section: _style_spec(
        ListToStringSpec, _parse_separators(section), section
    ),
    "number_to_string": lambda section: NumberToStringSpec(
        variants=_parse_variants(section),
        probability=section.optional("probability", float, 1.0),
    ),
    "card_dropout": lambda section: CardDropoutSpec(
        section.required("drop_probability", float), _parse_groups(section)
    ),
    "card_subsample": lambda section: CardSubsampleSpec(
        section.required("keep_fraction", float), _parse_groups(section)
    ),
    "duplicate_collapse": lambda section: DuplicateCollapseSpec(_parse_groups(section)),
}


def _parse_variants(section: ConfigSection) -> int:
    """A card-field mod entry's optional `variants:` int; UNBOUNDED_VARIANTS
    when absent. The mod's constructor checks the upper bound.

    Inputs: section (one mod entry). Output: int.
    Side effects: records variants as read.
    Exceptions: ValueError if variants is not an int.
    """
    return section.optional("variants", int, UNBOUNDED_VARIANTS)


def _style_spec(
    spec_class: type[StyleChoiceSpec[Any]],
    styles: tuple[Any, ...] | None,
    section: ConfigSection,
) -> ModSpec:
    """A content style spec of spec_class from its parsed styles and the
    entry's optional variants and probability.

    Inputs: spec_class, styles (already parsed; None: the mod's
        DEFAULT_STYLES), section (the mod entry).
    Output: ModSpec.
    Side effects: records variants and probability as read.
    Exceptions: ValueError for malformed variants or probability, or as
        the mod's constructor.
    """
    return spec_class(
        styles,
        variants=_parse_variants(section),
        probability=section.optional("probability", float, 1.0),
    )


def _parse_style_choices(
    section: ConfigSection, choices: type[_StyleT]
) -> tuple[_StyleT, ...] | None:
    """A content style mod entry's optional `styles:` list, each item one
    of choices' values; None when absent (the mod's DEFAULT_STYLES).

    Inputs: section (one mod entry), choices (the StrEnum).
    Output: tuple of members, in the order written, or None.
    Side effects: records styles as read.
    Exceptions: ValueError naming the item's location and the valid
        values, for a non-string or unknown item. (Emptiness and repeats
        are the mod's own checks.)
    """
    if not section.has("styles"):
        return None
    return section.required_choice_list("styles", choices)


def _parse_separators(section: ConfigSection) -> tuple[str, ...] | None:
    """A flatten_keys / list_to_string entry's optional `styles:` list of
    separator strings; None when absent (the mod's DEFAULT_STYLES).

    Inputs: section. Output: tuple[str, ...] or None.
    Side effects: records styles as read.
    Exceptions: ValueError if styles is not a list of str.
    """
    if not section.has("styles"):
        return None
    return section.required_str_list("styles")


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
        checked_scalar(item, int, f"{where}.{index}")
        for index, item in enumerate(section.required_list("groups"))
    )


def parse_default_augmentation(section: ConfigSection) -> AugmentationRates:
    """The `default_augmentation:` section -> AugmentationRates, rewriting
    every game default card-field mod's rates (src/dojos/
    augmentation_defaults.py); dojos named in `mods:` are not touched:

        default_augmentation: {probability: 0.1, variants: 4}

    Both keys are optional; an absent key keeps each spec's own value.

    Inputs: section ("default_augmentation", possibly empty).
    Output: AugmentationRates. Side effects: records the keys as read.
    Exceptions: ValueError for an unknown key, a wrongly typed value, or a
        rate no card-field mod accepts.

    Example:
        >>> parse_default_augmentation(top.optional_section("default_augmentation"))
        AugmentationRates(probability=0.1, variants=4)
    """
    probability: float | None = None
    variants: int | None = None
    if section.has("probability"):
        probability = section.required("probability", float)
    if section.has("variants"):
        variants = section.required("variants", int)
    section.reject_unread_keys()
    return construct_at(section.where, lambda: AugmentationRates(probability, variants))
