import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.seventeenlands.v2.csv_scanner import CsvScanner
from src.data_refinement.seventeenlands.v2.game_data_metrics.on_play_win_rate_delta_metric import (
    OnPlayWinRateDeltaMetric,
)
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

def _write_csv(path: Path, rows: list[tuple[str, str]]) -> None:
    # rows: (won, on_play), card always in deck
    header = (
        "expansion,event_type,won,on_play,"
        "deck_Bolt,opening_hand_Bolt,drawn_Bolt,tutored_Bolt,sideboard_Bolt\n"
    )
    lines = [header]
    for won, on_play in rows:
        lines.append(f"MSH,PremierDraft,{won},{on_play},1,0,0,0,0\n")
    path.write_text("".join(lines))

def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

class TestAccumulateAndFinalize:
    def test_computes_delta_between_on_play_and_on_draw(self, tmp_path: Path) -> None:
        binder = CardBinder()
        bolt = binder.add(_card("Bolt")).stored_card
        csv_path = tmp_path / "game_data.csv"
        _write_csv(
            csv_path,
            rows=[
                ("True", "True"),
                ("True", "True"),
                ("False", "True"),
                ("False", "False"),
                ("False", "False"),
            ],
        )
        output_path = tmp_path / "out.jsonl"
        metric = OnPlayWinRateDeltaMetric(binder, GameId.MTG, "MSH", "PremierDraft", output_path)

        CsvScanner(csv_path, chunk_size=100_000, metrics=[metric]).scan()

        rows = _read_jsonl(output_path)
        assert len(rows) == 1
        assert rows[0]["nocab_uuid"] == str(bolt.nocab_uuid)
        assert rows[0]["sample_size"] == 2  # min(3 on-play, 2 on-draw)
        assert rows[0]["value"] == 2 / 3 - 0.0  # 2/3 on-play win rate, 0/2 on-draw

    def test_card_missing_from_either_bucket_omitted(self, tmp_path: Path) -> None:
        binder = CardBinder()
        binder.add(_card("Bolt"))
        csv_path = tmp_path / "game_data.csv"
        _write_csv(csv_path, rows=[("True", "True")])  # only on-play observed
        output_path = tmp_path / "out.jsonl"
        metric = OnPlayWinRateDeltaMetric(binder, GameId.MTG, "MSH", "PremierDraft", output_path)

        CsvScanner(csv_path, chunk_size=100_000, metrics=[metric]).scan()

        assert _read_jsonl(output_path) == []

class TestDefaultOutputPath:
    def test_matches_default_output_dir_and_name(self) -> None:
        expected = (
            OnPlayWinRateDeltaMetric.DEFAULT_OUTPUT_DIR
            / "MSH.PremierDraft.on_play_win_rate_delta.jsonl"
        )
        assert OnPlayWinRateDeltaMetric.default_output_path("MSH", "PremierDraft") == expected

