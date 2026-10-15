"""A run config's `mods:` section -> per-dojo augmentation specs
(src/dojos/mods/mod_specs.py)."""

from typing import Any, Callable

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
from src.training.config_parsing.config_section import (
    ConfigSection,
    checked_scalar,
    construct_at,
    parse_by_kind,
)
from src.utils.drop_table import DropTable


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
        checked_scalar(item, int, f"{where}.{index}")
        for index, item in enumerate(section.required_list("groups"))
    )
