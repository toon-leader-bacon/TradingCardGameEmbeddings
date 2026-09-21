"""Group training examples into batches that respect a BatchBudget."""

from typing import Iterable, Iterator, List

from src.dojos.dojo import BatchBudget
from src.schema.type_hints import TrainingDatum, iter_cards


def group_by_budget(
    data: Iterable[TrainingDatum], budget: BatchBudget
) -> Iterator[List[TrainingDatum]]:
    """Greedily pack consecutive examples into groups within a budget.

    Inputs:
        data: examples in the order they should be batched.
        budget: max summed budget.cost_of over every card of a group.
    Output: iterator of non-empty example lists, order preserved; each
        group's cost is at most budget.max_cost.
    Side effects: none.
    Exceptions: ValueError if a single example alone costs more than
        budget.max_cost (the budget cannot be honored at all).

    Example:
        >>> [len(g) for g in group_by_budget(data, BatchBudget(4, lambda c: 1))]
        [2, 2, 1]  # for 2-card examples... 4 cost holds two of them
    """
    group: List[TrainingDatum] = []
    group_cost = 0
    for datum in data:
        cost = sum(budget.cost_of(card) for card in iter_cards(datum[0]))
        if cost > budget.max_cost:
            raise ValueError(
                f"One example costs {cost}, more than the batch budget "
                f"{budget.max_cost}"
            )
        if group and group_cost + cost > budget.max_cost:
            yield group
            group, group_cost = [], 0
        group.append(datum)
        group_cost += cost
    if group:
        yield group
