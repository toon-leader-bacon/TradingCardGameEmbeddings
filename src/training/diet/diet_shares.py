"""Each phase's expected diet: what share of its steps each dojo gets while
every dojo is active. Printed by `run_training.py` (also under --check) and
written into the run directory, so a weighting like "contrastive gets
half" is checked before any training."""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from src.dojos.dojo import Dojo
from src.training.diet.count_weighting import TrainCountCache
from src.training.diet.table_diet import as_table_diet, build_dojo_drop_table
from src.training.plan import DietRule, TrainingPlan


@dataclass(frozen=True)
class DietShare:
    """One dojo's expected share of one phase's steps.

    phase_name: the phase.
    dojo_name: the dojo.
    train_count: its TRAIN example count (0 means never drawn; see share
        for the effective chance, which a zero row weight also makes 0).
    share: expected fraction of the phase's steps, in [0, 1].
    """

    phase_name: str
    dojo_name: str
    train_count: int
    share: float


def plan_diet_shares(plan: TrainingPlan, dojos: Sequence[Dojo]) -> list[DietShare]:
    """Every phase's expected per-dojo shares (expected_dojo_shares), phase
    by phase in plan order, dojos in phase order.

    Inputs: plan, dojos (every dojo the plan's phases name, built).
    Output: list[DietShare].
    Side effects: reads each phase dojo's TRAIN count (logs an unreadable
        one; it counts 0).
    Exceptions: KeyError if a phase names a dojo not in dojos; ValueError
        if a phase has no drawable dojo.

    Example:
        >>> plan_diet_shares(config.plan, dojos)[0]
        DietShare(phase_name='frozen', dojo_name='contrastive.gwent', train_count=48213, share=1.0)
    """
    result: list[DietShare] = []
    by_name = {dojo.name: dojo for dojo in dojos}
    counter = TrainCountCache()

    # Each phase: count its dojos, then read the shares off its diet
    for phase in plan.phases:
        counts = {name: counter.count(by_name[name]) for name in phase.dojo_names}
        shares = expected_dojo_shares(phase.diet_rule, phase.dojo_names, counts)
        for name in phase.dojo_names:
            result.append(DietShare(phase.name, name, counts[name], shares[name]))
    return result


def expected_dojo_shares(
    rule: DietRule, dojo_names: Sequence[str], counts: Mapping[str, int]
) -> dict[str, float]:
    """Each dojo's expected share of a phase's steps while every dojo is
    active, read off the table build_dojo_drop_table makes (a flat rule as
    a one-row table, as_table_diet).

    Inputs: rule (the phase's DietRule), dojo_names (the phase's dojos),
        counts (TRAIN example count per dojo; a missing name counts 0).
    Output: dict dojo name -> share in [0, 1], in dojo_names order; shares
        sum to 1 (a dojo never drawn has 0).
    Side effects: none.
    Exceptions: ValueError if no dojo is drawable.

    Example:
        >>> expected_dojo_shares(Uniform(), ["a", "b"], {"a": 10, "b": 99})
        {'a': 0.5, 'b': 0.5}
    """
    result: dict[str, float] = {name: 0.0 for name in dojo_names}

    # One build for every rule: a flat rule is a one-row table
    built = build_dojo_drop_table(as_table_diet(rule, dojo_names), counts)
    if built.table is None:
        raise ValueError("no dojo in the diet is drawable")

    # Read the shares off the built table
    result.update(built.table.outcome_probabilities())
    return result


def format_diet_shares(shares: Sequence[DietShare]) -> list[str]:
    """One line per share, for printing: "[phase]  12.50%  dojo (train N)".

    Inputs: shares. Output: list[str]. Side effects: none.
    Exceptions: none.

    Example:
        >>> format_diet_shares([DietShare("frozen", "a", 10, 0.5)])
        ['[frozen]  50.00%  a (train 10)']
    """
    return [
        f"[{share.phase_name}] {share.share:7.2%}  {share.dojo_name} "
        f"(train {share.train_count})"
        for share in shares
    ]


def write_diet_shares_csv(shares: Sequence[DietShare], path: Path) -> None:
    """Write shares as a CSV (phase, dojo, train_count, share).

    Inputs: shares, path (its directory must exist).
    Output: none. Side effects: writes path (overwrites).
    Exceptions: OSError on a write failure.

    Example:
        >>> write_diet_shares_csv(shares, run_directory / "diet_shares.csv")
    """
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["phase", "dojo", "train_count", "share"])
        # One row per (phase, dojo)
        for share in shares:
            writer.writerow(
                [share.phase_name, share.dojo_name, share.train_count, share.share]
            )
