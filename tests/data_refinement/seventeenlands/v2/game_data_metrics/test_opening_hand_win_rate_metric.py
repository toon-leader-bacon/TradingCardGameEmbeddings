import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.seventeenlands.v2.csv_scanner import CsvScanner
from src.data_refinement.seventeenlands.v2.game_data_metrics.opening_hand_win_rate_metric import (
    OpeningHandWinRateMetric,
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

def _write_csv(path: Path, rows: list[tuple[str, int]]) -> None:
    header = (
        "expansion,event_type,won,"
        "deck_Bolt,opening_hand_Bolt,drawn_Bolt,tutored_Bolt,sideboard_Bolt\n"
    )
    lines = [header]
    for won, opening_hand in rows:
        lines.append(f"MSH,PremierDraft,{won},0,{opening_hand},0,0,0\n")
    path.write_text("".join(lines))

def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

class TestAccumulateAndFinalize:
    def test_computes_wins_and_games_when_in_opening_hand(self, tmp_path: Path) -> None:
        binder = CardBinder()
        bolt = binder.add(_card("Bolt")).stored_card
        csv_path = tmp_path / "game_data.csv"
        _write_csv(csv_path, rows=[("True", 1), ("False", 1), ("True", 0)])
        output_path = tmp_path / "out.jsonl"
        metric = OpeningHandWinRateMetric(binder, GameId.MTG, "MSH", "PremierDraft", output_path)

        CsvScanner(csv_path, chunk_size=100_000, metrics=[metric]).scan()

        rows = _read_jsonl(output_path)
        assert len(rows) == 1
        assert rows[0]["nocab_uuid"] == str(bolt.nocab_uuid)
        assert rows[0]["sample_size"] == 2
        assert rows[0]["value"] == 0.5

class TestDefaultOutputPath:
    def test_matches_default_output_dir_and_name(self) -> None:
        expected = (
            OpeningHandWinRateMetric.DEFAULT_OUTPUT_DIR
            / "MSH.PremierDraft.opening_hand_win_rate.jsonl"
        )
        assert OpeningHandWinRateMetric.default_output_path("MSH", "PremierDraft") == expected

