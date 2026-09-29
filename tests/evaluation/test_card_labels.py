from pathlib import Path
from uuid import uuid4

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.evaluation.card_labels import (
    CardLabels,
    GameLabels,
    HoldoutTierLabels,
    MetricParquetLabels,
)
from src.evaluation.embedding_table import CardRow
from src.schema.game_id import GameId
from src.schema.holdout import HoldoutSpec


def _row(game: GameId = GameId.MTG) -> CardRow:
    return CardRow(uuid4(), game)


def _parquet(
    path: Path, rows: list[tuple[object, object]], **extra: list[object]
) -> Path:
    """A metric parquet with nocab_uuid / label columns (+ any extras),
    written by pandas as the metrics pipeline does."""
    frame = pd.DataFrame(
        {
            "nocab_uuid": [str(card_id) for card_id, _ in rows],
            "label": [label for _, label in rows],
            **extra,
        }
    )
    frame.to_parquet(path)
    return path


def _arrow_parquet(path: Path, card_ids: list[object], labels: pa.Array) -> Path:
    """A parquet written by pyarrow directly, with no pandas metadata: how
    a non-pandas producer's integer column with nulls looks."""
    table = pa.table(
        {"nocab_uuid": [str(card_id) for card_id in card_ids], "label": labels}
    )
    pq.write_table(table, path)
    return path


class TestMetricParquetLabels:
    def test_labels_come_from_the_parquet_as_strings(self, tmp_path: Path) -> None:
        a, b = _row(), _row()
        path = _parquet(
            tmp_path / "m.parquet",
            [(a.nocab_uuid, "rare"), (b.nocab_uuid, "common")],
            masked_field=[["rarity"], ["rarity"]],  # other columns are ignored
        )
        labels = MetricParquetLabels("rarity", [path])
        assert labels.name == "rarity"
        assert labels.label_of(a) == "rare"
        assert labels.label_of(b) == "common"

    def test_non_string_labels_are_converted_with_str(self, tmp_path: Path) -> None:
        a = _row()
        path = _parquet(tmp_path / "m.parquet", [(a.nocab_uuid, 3), (uuid4(), 5)])
        assert MetricParquetLabels("cost", [path]).label_of(a) == "3"

    def test_an_integer_column_with_a_null_keeps_integer_labels(
        self, tmp_path: Path
    ) -> None:
        # pandas would read this column as floats: 3 -> "3.0"
        a, b = _row(), _row()
        path = _arrow_parquet(
            tmp_path / "m.parquet",
            [a.nocab_uuid, b.nocab_uuid],
            pa.array([3, None], pa.int64()),
        )
        labels = MetricParquetLabels("cost", [path])
        assert labels.label_of(a) == "3"
        assert labels.label_of(b) is None

    def test_a_float_nan_label_leaves_the_card_unlabeled(self, tmp_path: Path) -> None:
        a, b = _row(), _row()
        path = _arrow_parquet(
            tmp_path / "m.parquet",
            [a.nocab_uuid, b.nocab_uuid],
            pa.array([0.5, float("nan")], pa.float32()),
        )
        labels = MetricParquetLabels("m", [path])
        assert labels.label_of(a) == "0.5"
        assert labels.label_of(b) is None

    def test_a_directory_is_not_a_parquet(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            MetricParquetLabels("m", [tmp_path])

    def test_an_unlisted_card_is_unlabeled(self, tmp_path: Path) -> None:
        path = _parquet(tmp_path / "m.parquet", [(uuid4(), "x")])
        assert MetricParquetLabels("m", [path]).label_of(_row()) is None

    def test_a_null_label_leaves_the_card_unlabeled(self, tmp_path: Path) -> None:
        a = _row()
        path = _parquet(tmp_path / "m.parquet", [(a.nocab_uuid, None)])
        assert MetricParquetLabels("m", [path]).label_of(a) is None

    def test_several_parquets_form_one_source(self, tmp_path: Path) -> None:
        mtg, gwent = _row(GameId.MTG), _row(GameId.GWENT)
        first = _parquet(tmp_path / "mtg.parquet", [(mtg.nocab_uuid, "common")])
        second = _parquet(tmp_path / "gwent.parquet", [(gwent.nocab_uuid, "epic")])
        labels = MetricParquetLabels("rarity", [first, second])
        assert labels.label_of(mtg) == "common"
        assert labels.label_of(gwent) == "epic"

    def test_a_card_twice_in_one_file_is_rejected(self, tmp_path: Path) -> None:
        card_id = uuid4()
        path = _parquet(tmp_path / "m.parquet", [(card_id, "a"), (card_id, "b")])
        with pytest.raises(ValueError, match="appears twice"):
            MetricParquetLabels("m", [path])

    def test_a_card_in_two_files_is_rejected(self, tmp_path: Path) -> None:
        card_id = uuid4()
        first = _parquet(tmp_path / "a.parquet", [(card_id, "a")])
        second = _parquet(tmp_path / "b.parquet", [(card_id, "a")])
        with pytest.raises(ValueError, match="b.parquet"):
            MetricParquetLabels("m", [first, second])

    def test_a_null_row_still_counts_as_a_duplicate(self, tmp_path: Path) -> None:
        card_id = uuid4()
        path = _parquet(tmp_path / "m.parquet", [(card_id, None), (card_id, "a")])
        with pytest.raises(ValueError, match="appears twice"):
            MetricParquetLabels("m", [path])

    @pytest.mark.parametrize("dropped", ["nocab_uuid", "label"])
    def test_a_missing_column_is_rejected(self, tmp_path: Path, dropped: str) -> None:
        path = tmp_path / "m.parquet"
        pd.DataFrame({"nocab_uuid": [str(uuid4())], "label": ["a"]}).drop(
            columns=[dropped]
        ).to_parquet(path)
        with pytest.raises(ValueError, match=dropped):
            MetricParquetLabels("m", [path])

    def test_a_bad_uuid_is_rejected_naming_the_file(self, tmp_path: Path) -> None:
        path = _parquet(tmp_path / "m.parquet", [("not-a-uuid", "a")])
        with pytest.raises(ValueError, match="m.parquet"):
            MetricParquetLabels("m", [path])

    def test_a_missing_file_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            MetricParquetLabels("m", [tmp_path / "absent.parquet"])

    @pytest.mark.parametrize("name, paths", [("", [Path("x.parquet")]), ("m", [])])
    def test_empty_name_or_paths_are_rejected(
        self, name: str, paths: list[Path]
    ) -> None:
        with pytest.raises(ValueError):
            MetricParquetLabels(name, paths)


def test_game_labels_are_the_game_value() -> None:
    labels = GameLabels()
    assert labels.name == "game"
    assert labels.label_of(_row(GameId.GWENT)) == "gwent"
    assert labels.label_of(_row(GameId.MTG)) == "mtg"


def test_holdout_tier_labels_agree_with_the_spec() -> None:
    spec = HoldoutSpec(
        seed=4, tier_ratios=(1, 1, 1), held_out_games=frozenset({GameId.GWENT})
    )
    labels = HoldoutTierLabels(spec)
    assert labels.name == "holdout_tier"
    rows = [_row() for _ in range(30)]
    assert [labels.label_of(row) for row in rows] == [
        spec.tier_of(row.nocab_uuid, row.source_game).value for row in rows
    ]
    assert labels.label_of(_row(GameId.GWENT)) == "validation"


def test_every_source_satisfies_the_protocol(tmp_path: Path) -> None:
    path = _parquet(tmp_path / "m.parquet", [(uuid4(), "a")])
    sources: list[CardLabels] = [
        MetricParquetLabels("m", [path]),
        GameLabels(),
        HoldoutTierLabels(HoldoutSpec.no_holdout()),
    ]
    assert [source.name for source in sources] == ["m", "game", "holdout_tier"]
