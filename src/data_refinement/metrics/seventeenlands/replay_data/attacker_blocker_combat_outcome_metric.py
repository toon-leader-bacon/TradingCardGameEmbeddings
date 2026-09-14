"""AttackerBlockerCombatOutcomeMetric - plans/replay_data_metrics.md's
"Attacker Group vs. Blocker Group -> Combat Outcome": for every
half-turn that had at least one attacker, group 1 = that half-turn's
creatures_attacked, group 2 = creatures_blocking, label = a signed
net-kill-count delta.

NEW SHAPE (second occurrence in this codebase, after
game_data.TutorTargetPoolMetric) - FAN-OUT STREAMING: one row fans out
to one output example PER QUALIFYING HALF-TURN, not per row. Written
the same way TutorTargetPoolMetric writes its fan-out: one
pa.Table.from_pydict() built from a row-count-many-wide dict, one
write_table() call per accumulate() call - except this fans out over
turns within a game, not over pool members.

No deck_box - this metric's identity is (draft_id, match_number,
game_number, actor, turn), never a deck_uuid.

SIGN CONVENTION SETTLED (plans/replay_data_metrics.md's "Open
questions" #2): net_kill_delta is attacker-favorable-positive -
len(the DEFENDING side's creatures_killed_combat this half-turn) minus
len(the ATTACKING side's - actor's own - creatures_killed_combat this
half-turn). A positive value means the attacker traded up (killed more
of the defender's creatures than it lost of its own that turn); a
negative value means the attack went badly for the attacker.
"""

from pathlib import Path
from typing import ClassVar, Iterable, Literal

import pyarrow as pa
import pyarrow.parquet as pq

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.metrics.seventeenlands.replay_data.replay_card_columns import (
    ReplayCardColumns,
)
from src.schema.game_id import GameId

_DEFAULT_OUTPUT_PATH = Path(
    "data/metrics/seventeenlands/replay_data/attacker_blocker_combat_outcome.parquet"
)

_OUTPUT_SCHEMA = pa.schema(
    [
        ("draft_id", pa.string()),
        ("match_number", pa.int64()),
        ("game_number", pa.int64()),
        ("actor", pa.string()),
        ("turn", pa.int64()),
        ("attacker_uuids", pa.list_(pa.string())),
        ("blocker_uuids", pa.list_(pa.string())),
        ("net_kill_delta", pa.int64()),
    ]
)


class AttackerBlockerCombatOutcomeMetric:
    """Every half-turn with an attacker -> (attacker group, blocker
    group, net-kill outcome), one row per qualifying half-turn.

    Satisfies the Metric[dict] Protocol (../../metric.py) structurally.
    """

    DEFAULT_OUTPUT_PATH: ClassVar[Path] = _DEFAULT_OUTPUT_PATH

    def __init__(
        self,
        card_binder: CardBinder,
        header: Iterable[str],
        source_game: GameId,
        output_path: Path | None = None,
    ) -> None:
        """
        Inputs:
            card_binder: registry to match this CSV's per-turn
                creatures_attacked/creatures_blocking/
                creatures_killed_combat Arena-ID cells against -
                assumed already fully populated for source_game. Never
                queried directly by this class - only through the
                ReplayCardColumns this constructor builds from it.
            header: this CSV's column names (e.g.
                pandas.read_csv(path, nrows=0).columns) - parsed once,
                here, into this instance's own ReplayCardColumns.
            source_game: which game's cards header names are matched
                against.
            output_path: overrides DEFAULT_OUTPUT_PATH when given.
        Output: none (constructor).
        Side effects: creates output_path's parent directories if
            missing; opens output_path for writing (truncating any
            existing file) via a pyarrow.parquet.ParquetWriter held
            open for the lifetime of this instance - callers MUST call
            finalize() when done, or the file is left incomplete.
        Exceptions: whatever pyarrow.parquet.ParquetWriter raises on
            failure to open output_path for writing.
        """
        self._replay_columns = ReplayCardColumns.from_header(
            header, card_binder, source_game
        )
        self._output_path = output_path or self.DEFAULT_OUTPUT_PATH
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = pq.ParquetWriter(self._output_path, _OUTPUT_SCHEMA)
        self._closed = False

    def accumulate(self, row: dict) -> None:
        """Convert one replay_data row into zero or more output rows
        (one per half-turn with at least one attacker) and write them
        immediately.

        Inputs:
            row: one replay_data CSV row, dict-like - see
                ../scanner.py's module docstring.
        Output: none.
        Side effects: writes one row per (actor, turn) half-turn on
            this row whose creatures_attacked is non-empty, to the open
            ParquetWriter - zero rows if the game had no attacks.
        Exceptions: implementation-defined (expected: none for a
            well-formed row - see ../scanner.py's isolation contract).

        Example:
            >>> metric = AttackerBlockerCombatOutcomeMetric(card_binder, header, GameId.MTG)
            >>> metric.accumulate(row)
            >>> metric.finalize()
        """
        half_turns = self._qualifying_half_turns(row)
        if not half_turns:
            return

        output_rows = self._output_rows(row, half_turns)
        self._writer.write_table(output_rows)

    def finalize(self) -> Path:
        """Close the underlying ParquetWriter.

        A true no-op relative to data - every row this instance will
        ever write was already written by accumulate(). Idempotent: a
        second call is a no-op.

        Inputs: none.
        Output: self._output_path.
        Side effects: closes the ParquetWriter opened in __init__, if
            not already closed.
        Exceptions: whatever ParquetWriter.close() raises.

        Example:
            >>> metric.finalize()
            PosixPath('data/metrics/seventeenlands/replay_data/attacker_blocker_combat_outcome.parquet')
        """
        if not self._closed:
            self._writer.close()
            self._closed = True
        return self._output_path

    def _qualifying_half_turns(
        self, row: dict
    ) -> list[tuple[Literal["user", "oppo"], int]]:
        """Every (actor, turn) half-turn on this row whose
        creatures_attacked is non-empty.

        Private helper - single consumer is accumulate().

        Inputs:
            row: one replay_data CSV row, dict-like.
        Output: every (actor, turn) pair, across both actors' own
            turn-number ranges, where
            self._replay_columns.arena_uuids(row[creatures_attacked
            column]) is non-empty.
        Side effects: none.
        Exceptions: none expected.
        """
        half_turns: list[tuple[Literal["user", "oppo"], int]] = []
        for actor in ReplayCardColumns.ACTORS:
            turn_numbers = (
                self._replay_columns.user_turn_numbers
                if actor == "user"
                else self._replay_columns.oppo_turn_numbers
            )
            for turn in turn_numbers:
                attackers = self._replay_columns.arena_uuids(
                    row[
                        ReplayCardColumns.turn_column(actor, turn, "creatures_attacked")
                    ]
                )
                if attackers:
                    half_turns.append((actor, turn))
        return half_turns

    def _net_kill_delta(
        self, row: dict, actor: Literal["user", "oppo"], turn: int
    ) -> int:
        """Signed net-kill-count delta for one qualifying half-turn.

        Private helper - single consumer is _output_rows(). See module
        docstring's "SIGN CONVENTION SETTLED" - positive favors the
        attacking side (actor).

        Inputs:
            row: one replay_data CSV row, dict-like.
            actor: which half-turn - "user" or "oppo" - the attacking
                side this half-turn.
            turn: that actor's own turn-number counter.
        Output: len(defending side's creatures_killed_combat) minus
            len(actor's own creatures_killed_combat), this half-turn.
        Side effects: none.
        Exceptions: none expected.
        """
        own_killed = self._replay_columns.arena_uuids(
            row[
                ReplayCardColumns.turn_column(
                    actor, turn, f"{actor}_creatures_killed_combat"
                )
            ]
        )
        defender = "oppo" if actor == "user" else "user"
        defender_killed = self._replay_columns.arena_uuids(
            row[
                ReplayCardColumns.turn_column(
                    actor, turn, f"{defender}_creatures_killed_combat"
                )
            ]
        )
        return len(defender_killed) - len(own_killed)

    def _output_rows(
        self, row: dict, half_turns: list[tuple[Literal["user", "oppo"], int]]
    ) -> pa.Table:
        """Build one multi-row pa.Table matching _OUTPUT_SCHEMA, one row
        per qualifying half-turn.

        Private helper - single consumer is accumulate().

        Inputs:
            row: the same row accumulate() received.
            half_turns: this row's already-computed
                _qualifying_half_turns(row).
        Output: a pa.Table matching _OUTPUT_SCHEMA with len(half_turns)
            rows: draft_id/match_number/game_number repeated per row,
            actor/turn/attacker_uuids/blocker_uuids/net_kill_delta one
            entry per half_turns member.
        Side effects: none.
        Exceptions: none expected.
        """
        row_count = len(half_turns)
        attacker_uuids = []
        blocker_uuids = []
        net_kill_deltas = []
        actors = []
        turns = []
        for actor, turn in half_turns:
            attackers = self._replay_columns.arena_uuids(
                row[ReplayCardColumns.turn_column(actor, turn, "creatures_attacked")]
            )
            blockers = self._replay_columns.arena_uuids(
                row[ReplayCardColumns.turn_column(actor, turn, "creatures_blocking")]
            )
            attacker_uuids.append([str(uuid) for uuid in attackers])
            blocker_uuids.append([str(uuid) for uuid in blockers])
            net_kill_deltas.append(self._net_kill_delta(row, actor, turn))
            actors.append(actor)
            turns.append(turn)

        return pa.Table.from_pydict(
            {
                "draft_id": [row["draft_id"]] * row_count,
                "match_number": [row["match_number"]] * row_count,
                "game_number": [row["game_number"]] * row_count,
                "actor": actors,
                "turn": turns,
                "attacker_uuids": attacker_uuids,
                "blocker_uuids": blocker_uuids,
                "net_kill_delta": net_kill_deltas,
            },
            schema=_OUTPUT_SCHEMA,
        )
