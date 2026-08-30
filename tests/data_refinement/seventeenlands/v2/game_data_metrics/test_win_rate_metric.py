import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.seventeenlands.game_data_metrics.metric_scanner import MetricScanner
from src.data_refinement.seventeenlands.game_data_metrics.metrics.win_rate.win_rate import (
    WinRateMetric as V1WinRateMetric,
)
from src.data_refinement.seventeenlands.v2.csv_scanner import CsvScanner
from src.data_refinement.seventeenlands.v2.game_data_metrics.win_rate_metric import WinRateMetric
from src.schema.card import GenericCard, Provenance
from src.schema.data_source import DataSource
from src.schema.game_id import GameId


def _card(name: str) -> GenericCard:
    return GenericCard(
        nocab_uuid=uuid4(),
        source_game=GameId.MTG,
        name=name,
        raw_content={},
        provenance=Provenance(
            data_source=DataSource.SCRYFALL,
            source_id=name,
            fetched_at=datetime.now(timezone.utc),
        ),
    )


def _write_csv(path: Path, rows: list[tuple[str, int, int]]) -> None:
    header = (
        "expansion,event_type,won,"
        "deck_Bolt,opening_hand_Bolt,drawn_Bolt,tutored_Bolt,sideboard_Bolt,"
        "deck_Bear,opening_hand_Bear,drawn_Bear,tutored_Bear,sideboard_Bear\n"
    )
    lines = [header]
    for won, bolt_count, bear_count in rows:
        lines.append(f"MSH,PremierDraft,{won},{bolt_count},0,0,0,0,{bear_count},0,0,0,0\n")
    path.write_text("".join(lines))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class TestAccumulateAndFinalize:
    def test_computes_wins_and_games(self, tmp_path: Path) -> None:
        binder = CardBinder()
        bolt = binder.add(_card("Bolt")).stored_card
        csv_path = tmp_path / "game_data.csv"
        _write_csv(csv_path, rows=[("True", 1, 0), ("True", 1, 0), ("False", 1, 0)])
        output_path = tmp_path / "out.jsonl"
        metric = WinRateMetric(binder, GameId.MTG, "MSH", "PremierDraft", output_path)

        CsvScanner(csv_path, chunk_size=100_000, metrics=[metric]).scan()

        rows = _read_jsonl(output_path)
        assert len(rows) == 1
        assert rows[0]["nocab_uuid"] == str(bolt.nocab_uuid)
        assert rows[0]["metric_name"] == "win_rate"
        assert rows[0]["sample_size"] == 3
        assert rows[0]["value"] == 2 / 3

    def test_card_never_in_any_deck_absent_from_results(self, tmp_path: Path) -> None:
        binder = CardBinder()
        bolt = binder.add(_card("Bolt")).stored_card
        bear = binder.add(_card("Bear")).stored_card
        csv_path = tmp_path / "game_data.csv"
        _write_csv(csv_path, rows=[("True", 1, 0), ("False", 1, 0)])
        output_path = tmp_path / "out.jsonl"
        metric = WinRateMetric(binder, GameId.MTG, "MSH", "PremierDraft", output_path)

        CsvScanner(csv_path, chunk_size=100_000, metrics=[metric]).scan()

        rows = _read_jsonl(output_path)
        uuids = {row["nocab_uuid"] for row in rows}
        assert uuids == {str(bolt.nocab_uuid)}
        assert str(bear.nocab_uuid) not in uuids

    def test_finalize_with_zero_resolved_cards_produces_a_valid_empty_file(
        self, tmp_path: Path
    ) -> None:
        binder = CardBinder()
        output_path = tmp_path / "out.jsonl"
        metric = WinRateMetric(binder, GameId.MTG, "MSH", "PremierDraft", output_path)

        result_path = metric.finalize()

        assert result_path == output_path
        assert _read_jsonl(output_path) == []


class TestDefaultOutputPath:
    def test_matches_default_output_dir_and_name(self) -> None:
        expected = WinRateMetric.DEFAULT_OUTPUT_DIR / "MSH.PremierDraft.win_rate.jsonl"
        assert WinRateMetric.default_output_path("MSH", "PremierDraft") == expected

    def test_varies_by_expansion_and_format(self) -> None:
        assert WinRateMetric.default_output_path(
            "MSH", "PremierDraft"
        ) != WinRateMetric.default_output_path("MSH", "TradDraft")


class TestMatchesV1Output:
    def test_v2_win_rate_matches_v1_win_rate_on_the_same_csv(self, tmp_path: Path) -> None:
        # The core validation this stress-test metric exists to
        # provide: v2's independently-resolving, independently-writing
        # WinRateMetric must compute the exact same per-card win rates
        # as the already-shipped, already-tested v1 WinRateMetric,
        # given the same input.
        binder = CardBinder()
        bolt = binder.add(_card("Bolt")).stored_card
        bear = binder.add(_card("Bear")).stored_card
        csv_path = tmp_path / "game_data.csv"
        _write_csv(
            csv_path,
            rows=[
                ("True", 1, 0),
                ("True", 1, 1),
                ("False", 1, 1),
                ("False", 0, 1),
            ],
        )

        # v1
        v1_metric = V1WinRateMetric(expansion="MSH", format_code="PremierDraft")
        v1_output_path = tmp_path / "v1_out.parquet"
        v1_scanner = MetricScanner(
            raw_csv_path=csv_path,
            card_binder=binder,
            metrics=[v1_metric],
            output_path=v1_output_path,
            checkpoint_path=tmp_path / "v1_checkpoint.json",
            source_game=GameId.MTG,
        )
        v1_scanner.scan()
        v1_results = {
            row["nocab_uuid"]: row["value"]
            for row in pd.read_parquet(v1_output_path).to_dict("records")
        }

        # v2
        v2_metric = WinRateMetric(
            binder, GameId.MTG, "MSH", "PremierDraft", tmp_path / "v2_out.jsonl"
        )
        CsvScanner(csv_path, chunk_size=100_000, metrics=[v2_metric]).scan()
        v2_results = {
            row["nocab_uuid"]: row["value"] for row in _read_jsonl(tmp_path / "v2_out.jsonl")
        }

        assert v1_results == v2_results
        assert v1_results[str(bolt.nocab_uuid)] == 2 / 3
        assert v1_results[str(bear.nocab_uuid)] == 1 / 3
