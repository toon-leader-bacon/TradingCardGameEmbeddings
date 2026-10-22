"""Group training examples into batches that respect a BatchBudget.

A padded batch holds examples x max_cards slots, and every slot costs
memory in the attention stack, real or padding. So a group is charged its
padded cost, not its summed cost: an outlier deck cannot pull a crowd of
small decks into one huge padded batch.
"""

from typing import Iterable, Iterator, List, Sequence

from src.dojos.dojo import BatchBudget
from src.schema.type_hints import LabelT, TrainingDatum, iter_cards


def padded_cost(example_costs: Sequence[int]) -> int:
    """What a batch of these examples costs once padded to the longest.

    Inputs: example_costs, one cost per example (each the summed
        budget.cost_of over its cards).
    Output: len(example_costs) * max(example_costs); 0 for no examples.
    Side effects: none. Exceptions: none.

    Example:
        >>> padded_cost([60, 600, 60])
        1800
    """
    return len(example_costs) * max(example_costs, default=0)


def group_by_budget(
    data: Iterable[TrainingDatum[LabelT]], budget: BatchBudget
) -> Iterator[List[TrainingDatum[LabelT]]]:
    """Greedily pack consecutive examples into groups within a padded budget.

    The grouping depends on the order of the examples; the bound holds in
    every order. An outlier first makes its group small; an outlier that
    would push a group over closes it and starts the next one.
    Inputs:
        data: examples in the order they should be batched.
        budget: bounds each group's padded_cost of its examples' costs.
    Output: iterator of non-empty example lists, order preserved; each
        group's padded_cost is at most budget.max_cost.
    Side effects: none.
    Exceptions: ValueError if a single example alone costs more than
        budget.max_cost (the budget cannot be honored at all).

    Example:
        >>> # unit card cost; examples of 2, 2, 1, 5 and 1 cards, budget 6
        >>> [len(g) for g in group_by_budget(data, BatchBudget(6, lambda c: 1))]
        [3, 1, 1]  # 3 x 2 = 6; then 5 alone (2 x 5 > 6); then 1
    """
    group: List[TrainingDatum[LabelT]] = []
    group_costs: List[int] = []
    for datum in data:
        # Validate: one example alone must fit
        cost = sum(budget.cost_of(card) for card in iter_cards(datum[0]))
        if cost > budget.max_cost:
            raise ValueError(
                f"One example costs {cost}, more than the batch budget "
                f"{budget.max_cost}"
            )
        # Close the group before the padded cost would go over
        if group and padded_cost(group_costs + [cost]) > budget.max_cost:
            yield group
            group, group_costs = [], []
        group.append(datum)
        group_costs.append(cost)
    if group:
        yield group
