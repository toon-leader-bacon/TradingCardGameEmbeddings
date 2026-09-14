"""DataConstructor for AttackerBlockerCombatOutcomeMetric - see
src/data_refinement/metrics/seventeenlands/replay_data/attacker_blocker_combat_outcome_metric.py.
"""

from typing import List

import pandas as pd

from src.data_refinement.card_binder.card_binder import CardBinder
from src.dojos.generic.data_constructors._uuid_resolution import _cards_for_uuids
from src.schema.type_hints import MultiGroupInput, TrainingDatum


class AttackerBlockerCombatOutcomeDataConstructor:
    """DataConstructor for AttackerBlockerCombatOutcomeMetric (see
    src/data_refinement/metrics/seventeenlands/replay_data/
    attacker_blocker_combat_outcome_metric.py) - two card groups
    (attackers, blockers) in, a signed net-kill-count delta out. Feeds
    MultiGroupRegressionDojo (src/dojos/generic/multi_group_regression/
    dojo.py): TrainingInput = MultiGroupInput = [attacker_cards,
    blocker_cards].

    **attacker group MUST stay index 0, blocker MUST stay index 1** -
    same reasoning as PoolConditionedPickDataConstructor's group
    ordering (see that class's docstring): src/schema/type_hints.py's
    input_shape_of() classifies shape by peeking group 0 only and
    raises ValueError on an empty list there. attacker_uuids is never
    empty on the metric's own output (a non-empty attacker group is the
    row's qualifying condition - see the metric's own
    _qualifying_half_turns()); blocker_uuids legitimately is (an
    unblocked attack).

    Unlike PackToPickChoiceSetDataConstructor/
    PoolConditionedPickDataConstructor, there's no position-in-a-list
    label here (the label is a plain scalar, not an index), so this
    reuses _cards_for_uuids() directly for both groups - no new
    "resolve strictly or skip the whole row" helper needed.
    """

    def __init__(self, card_binder: CardBinder) -> None:
        """
        Inputs:
            card_binder: registry to resolve each row's attacker_uuids
                and blocker_uuids against. Never written to.
        Output: none (constructor).
        Side effects: none.
        Exceptions: none.
        """
        self._card_binder = card_binder

    def build(self, chunk: pd.DataFrame) -> List[TrainingDatum]:
        """Convert a chunk of (attacker_uuids, blocker_uuids,
        net_kill_delta) rows into (MultiGroupInput, float) TrainingDatum
        pairs.

        Inputs:
            chunk: one chunk of rows from
                AttackerBlockerCombatOutcomeMetric's output parquet
                file.
        Output: one ([attacker_cards, blocker_cards], net_kill_delta)
            TrainingDatum per row whose attacker_uuids resolves to at
            least one card - mirrors DeckLabelDataConstructor's "empty
            result -> skip the row" convention, since an attacker group
            empty after resolution means every attacker uuid failed to
            resolve, not a valid training example. blocker_cards may be
            empty (a valid, expected row - an unblocked attack) via
            _cards_for_uuids(), which drops individual unresolvable
            blocker uuids rather than skipping the row - an empty
            blocker group is never itself a skip condition here.
            net_kill_delta is cast to float unconditionally (no NaN
            guard needed - the metric's own schema makes this column
            non-nullable, unlike OnPlayWinRateDeltaMetric's columns
            elsewhere in this package).
        Side effects: none.
        Exceptions: none expected (per-row failures are skipped, not
            raised - mirrors every other DataConstructor in this
            package).

        Example:
            >>> constructor = AttackerBlockerCombatOutcomeDataConstructor(card_binder)
            >>> constructor.build(chunk)
            [([[<GenericCard>], [<GenericCard>, <GenericCard>]], -1.0), ...]
        """
        results: List[TrainingDatum] = []

        # Attacker side must resolve to at least one card; blocker side
        # tolerates being empty - see build()'s own docstring.
        for _, row in chunk.iterrows():
            attacker_cards = _cards_for_uuids(self._card_binder, row["attacker_uuids"])
            if not attacker_cards:
                continue
            blocker_cards = _cards_for_uuids(self._card_binder, row["blocker_uuids"])
            group: MultiGroupInput = [attacker_cards, blocker_cards]
            results.append((group, float(row["net_kill_delta"])))

        return results
