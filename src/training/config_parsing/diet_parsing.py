"""A phase's `diet:` section -> DietRule (../plan.py): a flat rule
(uniform, proportional, temperature) or a table diet of dojo, dojo-group
and sub-table rows. See src/training/diet/README.md."""

import fnmatch
from typing import Callable, Literal

from src.training.config_parsing.config_section import ConfigSection, construct_at
from src.training.plan import (
    DietRule,
    DietTableRow,
    DojoGroupRow,
    DojoRow,
    FlatDietRule,
    Proportional,
    SubTableRow,
    TableDiet,
    Temperature,
    Uniform,
)

# Diet rules that take no parameters, by their config name
_PARAMETERLESS_DIETS: dict[str, Callable[[], FlatDietRule]] = {
    "uniform": Uniform,
    "proportional": Proportional,
}
_TEMPERATURE_DIET = "temperature"
_TABLE_DIET = "table"


def parse_diet(section: ConfigSection, run_dojos: tuple[str, ...]) -> DietRule:
    """A phase's "diet" -> DietRule: {"rule": "table", "table": [<rows>]}
    (see _parse_table_rows), or a flat rule (see _parse_flat_diet).

    Inputs: section (a phase's "diet"), run_dojos (the run's non-held-out
        dojos, which a table's name patterns match).
    Output: DietRule.
    Side effects: none.
    Exceptions: ValueError as _parse_flat_diet and _parse_table_rows.

    Example:
        >>> parse_diet(ConfigSection({"rule": "uniform"}, "diet"), ("a",))
        Uniform()
    """
    if section.has("rule") and section.required("rule", str) == _TABLE_DIET:
        rows = _parse_table_rows(section.required_section_list("table"), run_dojos)
        section.reject_unread_keys()
        return construct_at(section.where, lambda: TableDiet(rows))
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
            entry: DietTableRow = construct_at(
                row.where, lambda: DojoRow(weight, row.required("dojo", str))
            )
        elif kind == "dojos":
            entry = _parse_dojo_group_row(row, weight, run_dojos)
        else:
            sub_rows = _parse_table_rows(row.required_section_list("table"), run_dojos)
            entry = construct_at(row.where, lambda: SubTableRow(weight, sub_rows))
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
    run_dojos (matching_dojos), its optional "within" a flat rule
    (default uniform).

    Inputs: row, weight (already read), run_dojos.
    Output: DojoGroupRow.
    Side effects: marks "dojos" and "within" read.
    Exceptions: ValueError as matching_dojos and _parse_flat_diet.
    """
    dojo_names = matching_dojos(
        row.required_str_list("dojos"), run_dojos, row.location_of("dojos")
    )
    within: FlatDietRule = Uniform()
    if row.has("within"):
        within = _parse_flat_diet(row.required_section("within"))
    return construct_at(row.where, lambda: DojoGroupRow(weight, dojo_names, within))


def matching_dojos(
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
        result = construct_at(section.where, lambda: Temperature(alpha=alpha))
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
