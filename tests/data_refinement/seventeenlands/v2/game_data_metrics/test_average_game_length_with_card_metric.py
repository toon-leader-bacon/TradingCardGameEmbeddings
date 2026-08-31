import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.seventeenlands.v2.csv_scanner import CsvScanner
from src.data_refinement.seventeenlands.v2.game_data_metrics.average_game_length_with_card_metric import (  # noqa: E501 -- module path is one unbreakable dotted identifier
    AverageGameLengthWithCardMetric,
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

def _write_csv(path: Path, rows: list[tuple[int, int]]) -> None:
    # rows: (deck_count, num_turns)
    header = (
        "expansion,event_type,won,num_turns,"
        "deck_Bolt,opening_hand_Bolt,drawn_Bolt,tutored_Bolt,sideboard_Bolt\n"
    )
    lines = [header]
    for deck_count, num_turns in rows:
        lines.append(f"MSH,PremierDraft,True,{num_turns},{deck_count},0,0,0,0\n")
    path.write_text("".join(lines))

def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

class TestAccumulateAndFinalize:
    def test_averages_turns_over_included_games_only(self, tmp_path: Path) -> None:
        binder = CardBinder()
        bolt = binder.add(_card("Bolt")).stored_card
        csv_path = tmp_path / "game_data.csv"
        _write_csv(csv_path, rows=[(1, 10), (0, 99), (1, 8)])  # 99-turn game excluded
        output_path = tmp_path / "out.jsonl"
        metric = AverageGameLengthWithCardMetric(
            binder, GameId.MTG, "MSH", "PremierDraft", output_path
        )

        CsvScanner(csv_path, chunk_size=100_000, metrics=[metric]).scan()

        rows = _read_jsonl(output_path)
        assert len(rows) == 1
        assert rows[0]["nocab_uuid"] == str(bolt.nocab_uuid)
        assert rows[0]["sample_size"] == 2
        assert rows[0]["value"] == 9.0  # (10+8)/2

class TestDefaultOutputPath:
    def test_matches_default_output_dir_and_name(self) -> None:
        expected = (
            AverageGameLengthWithCardMetric.DEFAULT_OUTPUT_DIR
            / "MSH.PremierDraft.average_game_length_with_card.jsonl"
        )
        assert (
            AverageGameLengthWithCardMetric.default_output_path("MSH", "PremierDraft") == expected
        )

