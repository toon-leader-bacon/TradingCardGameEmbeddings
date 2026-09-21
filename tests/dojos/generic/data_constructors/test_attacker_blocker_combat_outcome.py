from datetime import datetime, timezone
from uuid import uuid4

import pandas as pd

from src.data_refinement.card_binder.card_binder import CardBinder
from src.dojos.generic.data_constructors import (
    AttackerBlockerCombatOutcomeDataConstructor,
)
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId


def _card(name: str, source_id: str) -> GenericCard:
    return GenericCard(
        nocab_uuid=uuid4(),
        source_game=GameId.SLAY_THE_SPIRE_2,
        name=name,
        raw_content={},
        provenance=Provenance(
            data_source=DataSource.SPIRE_CODEX,
            source_id=source_id,
            fetched_at=datetime.now(timezone.utc),
        ),
    )


class TestAttackerBlockerCombatOutcomeDataConstructorBuild:
    def test_builds_attacker_and_blocker_groups_with_float_label(self) -> None:
        binder = CardBinder()
        attacker, blocker = _card("Attacker", "attacker"), _card("Blocker", "blocker")
        binder.create(attacker)
        binder.create(blocker)
        chunk = pd.DataFrame(
            {
                "attacker_uuids": [[str(attacker.nocab_uuid)]],
                "blocker_uuids": [[str(blocker.nocab_uuid)]],
                "net_kill_delta": [-1],
            }
        )
        constructor = AttackerBlockerCombatOutcomeDataConstructor()

        result = constructor.build(chunk, binder)

        assert result == [([[attacker], [blocker]], -1.0)]

    def test_empty_blocker_uuids_produces_a_valid_row_with_an_empty_blocker_group(
        self,
    ) -> None:
        binder = CardBinder()
        attacker = _card("Attacker", "attacker")
        binder.create(attacker)
        chunk = pd.DataFrame(
            {
                "attacker_uuids": [[str(attacker.nocab_uuid)]],
                "blocker_uuids": [[]],
                "net_kill_delta": [2],
            }
        )
        constructor = AttackerBlockerCombatOutcomeDataConstructor()

        result = constructor.build(chunk, binder)

        assert result == [([[attacker], []], 2.0)]

    def test_skips_row_when_every_attacker_uuid_is_unresolvable(self) -> None:
        binder = CardBinder()
        chunk = pd.DataFrame(
            {
                "attacker_uuids": [[str(uuid4())]],
                "blocker_uuids": [[]],
                "net_kill_delta": [0],
            }
        )
        constructor = AttackerBlockerCombatOutcomeDataConstructor()

        result = constructor.build(chunk, binder)

        assert result == []

    def test_partially_unresolvable_attacker_uuids_are_dropped_but_row_is_kept(
        self,
    ) -> None:
        binder = CardBinder()
        attacker = _card("Attacker", "attacker")
        binder.create(attacker)
        chunk = pd.DataFrame(
            {
                "attacker_uuids": [[str(attacker.nocab_uuid), str(uuid4())]],
                "blocker_uuids": [[]],
                "net_kill_delta": [0],
            }
        )
        constructor = AttackerBlockerCombatOutcomeDataConstructor()

        result = constructor.build(chunk, binder)

        assert result == [([[attacker], []], 0.0)]

    def test_unresolvable_blocker_uuid_is_dropped_not_skipped(self) -> None:
        binder = CardBinder()
        attacker, blocker = _card("Attacker", "attacker"), _card("Blocker", "blocker")
        binder.create(attacker)
        binder.create(blocker)
        chunk = pd.DataFrame(
            {
                "attacker_uuids": [[str(attacker.nocab_uuid)]],
                "blocker_uuids": [[str(blocker.nocab_uuid), str(uuid4())]],
                "net_kill_delta": [1],
            }
        )
        constructor = AttackerBlockerCombatOutcomeDataConstructor()

        result = constructor.build(chunk, binder)

        assert result == [([[attacker], [blocker]], 1.0)]
