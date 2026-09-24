from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.data_refinement.metrics.version_metadata import (
    MetricVersionMetadata,
    metadata_from_schema,
    read_version_metadata,
    schema_with_version_metadata,
    write_dataframe_with_version_metadata,
)
from src.schema.game_id import GameId

_SCHEMA = pa.schema([("nocab_uuid", pa.string())])


class TestSchemaWithVersionMetadata:
    def test_round_trips_through_metadata_from_schema(self) -> None:
        metadata = MetricVersionMetadata(game=GameId.GWENT, card_binder_version="v1")

        schema = schema_with_version_metadata(_SCHEMA, metadata)

        assert metadata_from_schema(schema) == metadata

    def test_round_trips_requires_deck_box_when_true(self) -> None:
        metadata = MetricVersionMetadata(
            game=GameId.MTG, card_binder_version="v1", requires_deck_box=True
        )

        schema = schema_with_version_metadata(_SCHEMA, metadata)

        assert metadata_from_schema(schema) == metadata

    def test_preserves_existing_schema_metadata(self) -> None:
        schema = _SCHEMA.with_metadata({b"unrelated": b"kept"})
        metadata = MetricVersionMetadata(game=GameId.GWENT, card_binder_version="v1")

        new_schema = schema_with_version_metadata(schema, metadata)

        assert new_schema.metadata[b"unrelated"] == b"kept"
        assert metadata_from_schema(new_schema) == metadata

    def test_does_not_mutate_the_original_schema(self) -> None:
        metadata = MetricVersionMetadata(game=GameId.GWENT, card_binder_version="v1")

        schema_with_version_metadata(_SCHEMA, metadata)

        assert _SCHEMA.metadata is None


class TestMetadataFromSchema:
    def test_none_for_a_schema_with_no_metadata(self) -> None:
        assert metadata_from_schema(_SCHEMA) is None

    def test_none_for_a_schema_with_unrelated_metadata_only(self) -> None:
        schema = _SCHEMA.with_metadata({b"unrelated": b"value"})

        assert metadata_from_schema(schema) is None

    def test_requires_deck_box_is_false_when_not_given(self) -> None:
        metadata = MetricVersionMetadata(game=GameId.MTG, card_binder_version="v1")
        schema = schema_with_version_metadata(_SCHEMA, metadata)

        assert metadata_from_schema(schema).requires_deck_box is False

    def test_requires_deck_box_defaults_to_false_for_a_legacy_schema(self) -> None:
        # A schema stamped before requires_deck_box existed carries
        # game/card_binder_version but no requires_deck_box key at all -
        # must default to False, not raise or mis-parse.
        schema = _SCHEMA.with_metadata(
            {b"game": GameId.MTG.value.encode(), b"card_binder_version": b"v1"}
        )

        metadata = metadata_from_schema(schema)

        assert metadata.requires_deck_box is False


class TestReadVersionMetadata:
    def test_reads_what_write_dataframe_with_version_metadata_wrote(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "metric.parquet"
        metadata = MetricVersionMetadata(
            game=GameId.DOMINION, card_binder_version="v1", requires_deck_box=True
        )

        write_dataframe_with_version_metadata(
            pd.DataFrame({"nocab_uuid": ["a", "b"]}), path, metadata
        )

        assert read_version_metadata(path) == metadata

    def test_none_for_a_file_with_no_embedded_metadata(self, tmp_path: Path) -> None:
        path = tmp_path / "metric.parquet"
        pq.write_table(pa.Table.from_pydict({"nocab_uuid": ["a"]}), path)

        assert read_version_metadata(path) is None


class TestWriteDataframeWithVersionMetadata:
    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "metric.parquet"
        metadata = MetricVersionMetadata(game=GameId.MTG, card_binder_version="v1")

        write_dataframe_with_version_metadata(
            pd.DataFrame({"nocab_uuid": ["a"]}), path, metadata
        )

        assert path.exists()

    def test_writes_every_row(self, tmp_path: Path) -> None:
        path = tmp_path / "metric.parquet"
        metadata = MetricVersionMetadata(game=GameId.MTG, card_binder_version="v1")

        write_dataframe_with_version_metadata(
            pd.DataFrame({"nocab_uuid": ["a", "b", "c"]}), path, metadata
        )

        assert len(pd.read_parquet(path)) == 3
