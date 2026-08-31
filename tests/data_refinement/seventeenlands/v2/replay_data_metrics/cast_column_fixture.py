"""Shared CSV-header/row builder for CastEventScanner-based v2 replay tests.

CastEventScanner.find_cast_events() reads every <side>_turn_<N>_<family>
column unconditionally (no missing-column guard — see that module's own
docstring: "must include every <side>_turn_<N>_<family> column"), so any
test driving a metric built on it needs an identically-shaped ~240-column
header regardless of how few turns that test actually populates.

Extracted here rather than duplicated across test_average_turn_cast_metric.py,
test_average_turns_remaining_post_cast_metric.py, and
test_opponent_response_rate_metric.py — those three files were authored
together with foreknowledge of the duplication (same shape, same
correctness-critical column-naming rules), so this follows this
session's "extract now" precedent (see replay_data_turn_extraction_utils.py,
metric_result_writer.py) rather than "rule of three, wait for organic
accretion". Every other test-file helper in this directory (_card,
_read_jsonl, etc.) stays duplicated per-file, per this project's usual
test convention — this one is an exception because the logic itself
(not just boilerplate) would silently drift if reimplemented three times.
"""

from pathlib import Path

_SIDES = ("user", "oppo")
_MAX_TURN = 30


def cast_column_names() -> list[str]:
    """Every <side>_turn_<N>_<family> column CastEventScanner reads.

    Inputs: none.
    Output: the ~240 column names CastEventScanner.find_cast_events()
        unconditionally accesses, for _MAX_TURN turns and both sides,
        in a stable order.
    Side effects: none.
    Exceptions: none.
    """
    columns: list[str] = []
    for turn in range(1, _MAX_TURN + 1):
        for side in _SIDES:
            other = "oppo" if side == "user" else "user"
            columns.append(f"{side}_turn_{turn}_creatures_cast")
            columns.append(f"{side}_turn_{turn}_non_creatures_cast")
            columns.append(f"{side}_turn_{turn}_{side}_instants_sorceries_cast")
            columns.append(f"{other}_turn_{turn}_{side}_instants_sorceries_cast")
    return list(dict.fromkeys(columns))


def write_cast_csv(path: Path, base_columns: list[str], rows: list[dict[str, object]]) -> None:
    """Write a replay CSV with base_columns plus every cast-scan column.

    Inputs:
        path: where to write the CSV.
        base_columns: non-cast columns this test also needs (e.g.
            "num_turns"), in the order they should appear first.
        rows: one dict per row, mapping column name -> cell value.
            Any column (base or cast) missing from a row's dict is
            written as an empty cell.
    Output: none.
    Side effects: writes path.
    Exceptions: none expected.
    """
    all_columns = base_columns + cast_column_names()
    lines = [",".join(all_columns) + "\n"]
    for row in rows:
        lines.append(",".join(str(row.get(column, "")) for column in all_columns) + "\n")
    path.write_text("".join(lines))
