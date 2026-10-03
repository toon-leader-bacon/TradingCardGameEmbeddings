"""Tests for slice_file.py: SeventeenLandsSliceFile, SliceFingerprint and
finished_count_table, over hand-written partitions under tmp_path."""

import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.data_refinement.metrics.seventeenlands.count_table import write_count_table
from src.data_refinement.metrics.seventeenlands.data_slice import (
    SeventeenLandsSlice,
)
from src.data_refinement.metrics.seventeenlands.game_data.game_deck_label_metrics import (
    DeckWinPredictionMetric,
)
from src.data_refinement.metrics.seventeenlands.game_data.game_length_association_metric import (
    GameLengthAssociationMetric,
)
from src.data_refinement.metrics.seventeenlands.game_data.tutor_target_rate_metric import (
    TutorTargetRateMetric,
)
from src.data_refinement.metrics.seventeenlands.partition import (
    SeventeenLandsPartition,
)
from src.data_refinement.metrics.seventeenlands.slice_file import (
    SeventeenLandsSliceFile,
    SliceFingerprint,
    finished_count_table,
)
from src.data_refinement.metrics.version_metadata import (
    MetricVersionMetadata,
    read_version_metadata,
    schema_with_version_metadata,
)
from src.data_retrieval.seventeenlands.refs import DataType, Expansion, format_code
from src.schema.game_id import GameId

VERSION = MetricVersionMetadata(game=GameId.MTG, card_binder_version="v1")
KTK_SEALED = (Expansion.KTK, format_code.Sealed)
MSH_PREMIER = (Expansion.MSH, format_code.PremierDraft)
PREMIER_ONLY = SeventeenLandsSlice(formats=frozenset({format_code.PremierDraft}))
# A whole-second mtime: every filesystem stores it exactly
_SENTINEL_NS = 1_000_000_000 * 1_000_000_000


def _partition_path(
    root: Path, stem: str, where: tuple[Expansion, format_code]
) -> Path:
    expansion, fmt = where
    return SeventeenLandsPartition(DataType.GAME, stem, expansion, fmt).path(root)


def _write_tutor(
    root: Path,
    where: tuple[Expansion, format_code],
    rows: dict[str, tuple[float, float]],
    version: MetricVersionMetadata = VERSION,
) -> Path:
    """One TutorTargetRateMetric partition: {card: (in_deck, tutored)}."""
    return write_count_table(
        _partition_path(root, TutorTargetRateMetric.OUTPUT_STEM, where),
        TutorTargetRateMetric,
        {"nocab_uuid": list(rows)},
        {
            "in_deck": np.array([v[0] for v in rows.values()], np.float64),
            "tutored": np.array([v[1] for v in rows.values()], np.float64),
        },
        version,
    )


def _write_deck_wins(
    root: Path, where: tuple[Expansion, format_code], decks: list[str]
) -> Path:
    """One DeckWinPredictionMetric partition, one won game per deck."""
    path = _partition_path(root, DeckWinPredictionMetric.OUTPUT_STEM, where)
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema(
        [
            ("draft_id", pa.string()),
            ("match_number", pa.int64()),
            ("game_number", pa.int64()),
            ("deck_uuid", pa.string()),
            ("won", pa.bool_()),
        ]
    )
    table = pa.table(
        {
            "draft_id": [f"d{i}" for i in range(len(decks))],
            "match_number": [1] * len(decks),
            "game_number": [1] * len(decks),
            "deck_uuid": decks,
            "won": [True] * len(decks),
        },
        schema=schema,
    )
    pq.write_table(table.cast(schema_with_version_metadata(schema, VERSION)), path)
    return path


def _rows_by_card(path: Path) -> dict[str, dict]:
    return {row.pop("nocab_uuid"): row for row in pq.read_table(path).to_pylist()}


class TestCountTableSlices:
    def test_all_slice_sums_counts_across_partitions(self, tmp_path: Path) -> None:
        _write_tutor(tmp_path, KTK_SEALED, {"a": (2, 1), "b": (1, 0)})
        _write_tutor(tmp_path, MSH_PREMIER, {"a": (2, 0)})

        path = SeventeenLandsSliceFile(TutorTargetRateMetric, tmp_path).build(
            SeventeenLandsSlice()
        )

        assert _rows_by_card(path) == {
            "a": {"tutor_target_rate": 0.25, "sample_count": 4},
            "b": {"tutor_target_rate": 0.0, "sample_count": 1},
        }
        assert path == (
            tmp_path / "seventeenlands/game_data/slices/tutor_target_rate.all.parquet"
        )

    def test_a_filtered_slice_reads_only_its_partitions(self, tmp_path: Path) -> None:
        _write_tutor(tmp_path, KTK_SEALED, {"a": (2, 2)})
        _write_tutor(tmp_path, MSH_PREMIER, {"a": (2, 0)})

        path = SeventeenLandsSliceFile(TutorTargetRateMetric, tmp_path).build(
            PREMIER_ONLY
        )

        assert _rows_by_card(path) == {
            "a": {"tutor_target_rate": 0.0, "sample_count": 2}
        }
        assert path.name == "tutor_target_rate.formats-PremierDraft.parquet"

    def test_the_baseline_is_the_slices_own(self, tmp_path: Path) -> None:
        # Partition means: KTK 10 turns, MSH 20 turns; the slice mean is 15
        for where, card_turns, baseline in [
            (KTK_SEALED, 12.0, [2.0, 20.0]),
            (MSH_PREMIER, 18.0, [2.0, 40.0]),
        ]:
            write_count_table(
                _partition_path(
                    tmp_path, GameLengthAssociationMetric.OUTPUT_STEM, where
                ),
                GameLengthAssociationMetric,
                {"nocab_uuid": ["a"]},
                {"games": np.array([1.0]), "value_sum": np.array([card_turns])},
                VERSION,
                baseline=np.array(baseline),
            )
        slice_file = SeventeenLandsSliceFile(GameLengthAssociationMetric, tmp_path)

        everything = _rows_by_card(slice_file.build(SeventeenLandsSlice()))
        premier = _rows_by_card(slice_file.build(PREMIER_ONLY))

        assert everything["a"]["game_length_association"] == pytest.approx(0.0)
        assert premier["a"]["game_length_association"] == pytest.approx(-2.0)

    def test_the_built_file_carries_the_partitions_version(
        self, tmp_path: Path
    ) -> None:
        _write_tutor(tmp_path, KTK_SEALED, {"a": (1, 1)})

        path = SeventeenLandsSliceFile(TutorTargetRateMetric, tmp_path).build(
            SeventeenLandsSlice()
        )

        assert read_version_metadata(path) == VERSION


class TestRowStreamSlices:
    def test_concatenates_partitions_with_set_and_format(self, tmp_path: Path) -> None:
        _write_deck_wins(tmp_path, KTK_SEALED, ["d1", "d2"])
        _write_deck_wins(tmp_path, MSH_PREMIER, ["d3"])

        path = SeventeenLandsSliceFile(DeckWinPredictionMetric, tmp_path).build(
            SeventeenLandsSlice()
        )

        table = pq.read_table(path)
        assert table.column("deck_uuid").to_pylist() == ["d1", "d2", "d3"]
        assert table.column("set").to_pylist() == ["KTK", "KTK", "MSH"]
        assert table.column("format").to_pylist() == [
            "Sealed",
            "Sealed",
            "PremierDraft",
        ]
        assert read_version_metadata(path) == VERSION

    def test_a_filtered_row_stream(self, tmp_path: Path) -> None:
        _write_deck_wins(tmp_path, KTK_SEALED, ["d1"])
        _write_deck_wins(tmp_path, MSH_PREMIER, ["d3"])

        path = SeventeenLandsSliceFile(DeckWinPredictionMetric, tmp_path).build(
            PREMIER_ONLY
        )

        assert pq.read_table(path).column("deck_uuid").to_pylist() == ["d3"]


class TestCaching:
    def test_an_unchanged_slice_is_reused(self, tmp_path: Path) -> None:
        _write_tutor(tmp_path, KTK_SEALED, {"a": (1, 1)})
        slice_file = SeventeenLandsSliceFile(TutorTargetRateMetric, tmp_path)
        path = slice_file.build(SeventeenLandsSlice())
        os.utime(path, ns=(_SENTINEL_NS, _SENTINEL_NS))

        slice_file.build(SeventeenLandsSlice())

        assert path.stat().st_mtime_ns == _SENTINEL_NS

    def test_a_changed_partition_rebuilds(self, tmp_path: Path) -> None:
        _write_tutor(tmp_path, KTK_SEALED, {"a": (1, 1)})
        slice_file = SeventeenLandsSliceFile(TutorTargetRateMetric, tmp_path)
        slice_file.build(SeventeenLandsSlice())

        _write_tutor(tmp_path, KTK_SEALED, {"a": (4, 1), "b": (1, 0)})
        path = slice_file.build(SeventeenLandsSlice())

        assert _rows_by_card(path)["a"]["sample_count"] == 4

    def test_a_new_label_version_rebuilds(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_tutor(tmp_path, KTK_SEALED, {"a": (1, 1)})
        slice_file = SeventeenLandsSliceFile(TutorTargetRateMetric, tmp_path)
        path = slice_file.build(SeventeenLandsSlice())
        os.utime(path, ns=(_SENTINEL_NS, _SENTINEL_NS))

        monkeypatch.setattr(TutorTargetRateMetric, "LABEL_VERSION", 99)
        slice_file.build(SeventeenLandsSlice())

        assert path.stat().st_mtime_ns != _SENTINEL_NS
        fingerprint = SliceFingerprint.read_from(path)
        assert fingerprint is not None and fingerprint.label_version == 99

    def test_a_missing_file_has_no_fingerprint(self, tmp_path: Path) -> None:
        assert SliceFingerprint.read_from(tmp_path / "missing.parquet") is None


class TestErrors:
    def test_no_matching_partition(self, tmp_path: Path) -> None:
        _write_tutor(tmp_path, KTK_SEALED, {"a": (1, 1)})

        with pytest.raises(ValueError, match="no tutor_target_rate partitions"):
            SeventeenLandsSliceFile(TutorTargetRateMetric, tmp_path).build(PREMIER_ONLY)

    def test_partitions_from_two_binder_versions(self, tmp_path: Path) -> None:
        _write_tutor(tmp_path, KTK_SEALED, {"a": (1, 1)})
        other = MetricVersionMetadata(game=GameId.MTG, card_binder_version="v2")
        _write_tutor(tmp_path, MSH_PREMIER, {"a": (1, 1)}, version=other)

        with pytest.raises(ValueError, match="was built from"):
            SeventeenLandsSliceFile(TutorTargetRateMetric, tmp_path).build(
                SeventeenLandsSlice()
            )

    def test_finished_count_table_needs_a_table(self) -> None:
        with pytest.raises(ValueError, match="no count tables"):
            finished_count_table(TutorTargetRateMetric, [])

    def test_finished_count_table_rejects_unexpected_baseline_rows(self) -> None:
        table = pa.table(
            {
                "nocab_uuid": pa.array([None], pa.string()),
                "in_deck": [1.0],
                "tutored": [1.0],
            }
        )

        with pytest.raises(ValueError, match="baseline"):
            finished_count_table(TutorTargetRateMetric, [table])
