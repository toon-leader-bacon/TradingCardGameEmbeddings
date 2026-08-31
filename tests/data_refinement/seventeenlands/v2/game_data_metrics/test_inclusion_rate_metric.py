import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.seventeenlands.v2.csv_scanner import CsvScanner
from src.data_refinement.seventeenlands.v2.game_data_metrics.inclusion_rate_metric import (
    InclusionRateMetric,
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

def _write_csv(path: Path, deck_counts: list[int]) -> None:
    header = (
        "expansion,event_type,won,"
        "deck_Bolt,opening_hand_Bolt,drawn_Bolt,tutored_Bolt,sideboard_Bolt\n"
    )
    lines = [header]
    for count in deck_counts:
        lines.append(f"MSH,PremierDraft,True,{count},0,0,0,0\n")
    path.write_text("".join(lines))

def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

class TestAccumulateAndFinalize:
    def test_inclusion_rate_over_all_games_processed(self, tmp_path: Path) -> None:
        binder = CardBinder()
        bolt = binder.add(_card("Bolt")).stored_card
        csv_path = tmp_path / "game_data.csv"
        _write_csv(csv_path, deck_counts=[1, 0, 1, 0])  # denominator is all 4 games
        output_path = tmp_path / "out.jsonl"
        metric = InclusionRateMetric(binder, GameId.MTG, "MSH", "PremierDraft", output_path)

        CsvScanner(csv_path, chunk_size=100_000, metrics=[metric]).scan()

        rows = _read_jsonl(output_path)
        assert len(rows) == 1
        assert rows[0]["nocab_uuid"] == str(bolt.nocab_uuid)
        assert rows[0]["sample_size"] == 4
        assert rows[0]["value"] == 0.5  # 2 of 4 games

class TestDefaultOutputPath:
    def test_matches_default_output_dir_and_name(self) -> None:
        expected = InclusionRateMetric.DEFAULT_OUTPUT_DIR / "MSH.PremierDraft.inclusion_rate.jsonl"
        assert InclusionRateMetric.default_output_path("MSH", "PremierDraft") == expected

