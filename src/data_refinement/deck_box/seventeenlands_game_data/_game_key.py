"""One game_data row's position within its draft, and its provenance round trip.

Single consumer: src/data_refinement/deck_box/seventeenlands_game_data/extraction_stage.py.
See that module's docstring (CANONICAL GAME section) for why the lowest
_GameKey within a draft is the one kept.
"""

from typing import NamedTuple

from src.schema.card import Provenance


class _GameKey(NamedTuple):
    """One game's position within its draft; the lowest one is canonical.

    Field order is the comparison order: NamedTuple compares as a tuple,
    so build_index decides first, then match_number, then game_number
    (see extraction_stage.py's module docstring's CANONICAL GAME
    section). Owns both directions of the provenance.source_id round
    trip (source_id_for() and from_provenance()), so the written and
    parsed formats can't drift.
    """

    build_index: int
    match_number: int
    game_number: int

    @staticmethod
    def from_row(row: dict) -> "_GameKey":
        """Parse one game_data row's build_index/match_number/game_number.

        Inputs:
            row: one game_data CSV row, dict-like. pandas may hand the
                numbers over as floats (e.g. 1.0) when a chunk holds a
                NaN; int() normalizes them. match_number counts as 0
                when the row has no such key, which (since pandas gives
                every row every header column) means the whole file
                lacks the column - see extraction_stage.py's module
                docstring's CANONICAL GAME section.
        Output: that row's _GameKey.
        Side effects: none.
        Exceptions: raises KeyError if build_index or game_number is
            missing, and ValueError if any present value is NaN.
        """
        return _GameKey(
            int(row["build_index"]),
            int(row.get("match_number", 0)),
            int(row["game_number"]),
        )

    def source_id_for(self, draft_id: str) -> str:
        """Format the provenance.source_id that from_provenance() parses.

        Inputs:
            draft_id: the game's draft_id.
        Output: f"{draft_id}:{build_index}:{match_number}:{game_number}",
            e.g. "d1:0:1:2".
        Side effects: none.
        Exceptions: none.
        """
        return f"{draft_id}:{self.build_index}:{self.match_number}:{self.game_number}"

    @staticmethod
    def from_provenance(provenance: Provenance | None) -> "_GameKey | None":
        """Parse the game key a stored deck's provenance.source_id records.

        Inputs:
            provenance: a stored deck's provenance, whose source_id is
                f"{draft_id}:{build_index}:{match_number}:{game_number}".
        Output: that game's _GameKey, or None when provenance is None
            or source_id doesn't have that shape (a deck this stage
            didn't write). Callers treat None as "replace it".
        Side effects: none.
        Exceptions: none.
        """
        if provenance is None:
            return None
        # rsplit: the three numbers are always the last three fields,
        # even if a draft_id ever contained a colon.
        fields = provenance.source_id.rsplit(":", 3)
        numbers = fields[1:]
        if len(fields) != 4 or not all(number.isdigit() for number in numbers):
            return None
        return _GameKey(*(int(number) for number in numbers))
