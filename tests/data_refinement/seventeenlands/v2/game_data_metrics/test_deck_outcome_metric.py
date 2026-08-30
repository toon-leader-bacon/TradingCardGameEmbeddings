import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.seventeenlands.v2.csv_scanner import CsvScanner
from src.data_refinement.seventeenlands.v2.game_data_metrics.deck_outcome_metric import (
    DeckOutcomeMetric,
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
    def test_reconstructs_decks_with_duplicates(self, tmp_path: Path) -> None:
        binder = CardBinder()
        bolt = binder.add(_card("Bolt")).stored_card
        bear = binder.add(_card("Bear")).stored_card
        csv_path = tmp_path / "game_data.csv"
        _write_csv(csv_path, rows=[("True", 2, 1), ("False", 0, 3)])
        output_path = tmp_path / "out.jsonl"
        metric = DeckOutcomeMetric(binder, GameId.MTG, "MSH", "PremierDraft", output_path)

        result_path = CsvScanner(csv_path, chunk_size=100_000, metrics=[metric]).scan()[0]

        assert result_path == output_path
        rows = _read_jsonl(output_path)
        assert len(rows) == 2
        assert sorted(rows[0]["deck_card_uuids"]) == sorted(
            [str(bolt.nocab_uuid)] * 2 + [str(bear.nocab_uuid)] * 1
        )
        assert rows[0]["won"] is True
        assert rows[0]["expansion"] == "MSH"
        assert rows[0]["format"] == "PremierDraft"
        assert rows[1]["deck_card_uuids"] == [str(bear.nocab_uuid)] * 3
        assert rows[1]["won"] is False

    def test_unresolved_card_excluded_but_row_still_emitted(self, tmp_path: Path) -> None:
        binder = CardBinder()  # neither card registered
        csv_path = tmp_path / "game_data.csv"
        _write_csv(csv_path, rows=[("True", 1, 1)])
        output_path = tmp_path / "out.jsonl"
        metric = DeckOutcomeMetric(binder, GameId.MTG, "MSH", "PremierDraft", output_path)

        CsvScanner(csv_path, chunk_size=100_000, metrics=[metric]).scan()

        rows = _read_jsonl(output_path)
        assert len(rows) == 1
        assert rows[0]["deck_card_uuids"] == []
        assert rows[0]["won"] is True

    def test_row_with_no_tracked_cards_still_produces_a_line(self, tmp_path: Path) -> None:
        binder = CardBinder()
        binder.add(_card("Bolt"))
        binder.add(_card("Bear"))
        csv_path = tmp_path / "game_data.csv"
        _write_csv(csv_path, rows=[("False", 0, 0)])
        output_path = tmp_path / "out.jsonl"
        metric = DeckOutcomeMetric(binder, GameId.MTG, "MSH", "PremierDraft", output_path)

        CsvScanner(csv_path, chunk_size=100_000, metrics=[metric]).scan()

        rows = _read_jsonl(output_path)
        assert len(rows) == 1
        assert rows[0]["deck_card_uuids"] == []

    def test_finalize_with_no_accumulate_calls_produces_a_valid_empty_file(
        self, tmp_path: Path
    ) -> None:
        binder = CardBinder()
        output_path = tmp_path / "out.jsonl"
        metric = DeckOutcomeMetric(binder, GameId.MTG, "MSH", "PremierDraft", output_path)

        result_path = metric.finalize()

        assert result_path == output_path
        assert output_path.exists()
        assert _read_jsonl(output_path) == []


class TestDefaultOutputPath:
    def test_matches_default_output_dir_and_name(self) -> None:
        expected = DeckOutcomeMetric.DEFAULT_OUTPUT_DIR / "MSH.PremierDraft.deck_outcomes.jsonl"
        assert DeckOutcomeMetric.default_output_path("MSH", "PremierDraft") == expected

    def test_varies_by_expansion_and_format(self) -> None:
        assert DeckOutcomeMetric.default_output_path(
            "MSH", "PremierDraft"
        ) != DeckOutcomeMetric.default_output_path("MSH", "TradDraft")
