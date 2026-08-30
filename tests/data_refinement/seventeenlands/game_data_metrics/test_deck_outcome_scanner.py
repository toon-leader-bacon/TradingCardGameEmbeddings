import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.seventeenlands.game_data_metrics.deck_outcome_scanner import (
    DeckOutcomeScanner,
    DeckOutcomeScanResult,
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
    """Write a minimal game_data CSV with two tracked cards, Bolt/Bear.

    Each row is (won, bolt_deck_count, bear_deck_count).
    """
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


class TestScan:
    def test_fresh_scan_reconstructs_decks_with_duplicates(self, tmp_path: Path) -> None:
        binder = CardBinder()
        bolt = binder.add(_card("Bolt")).stored_card
        bear = binder.add(_card("Bear")).stored_card
        csv_path = tmp_path / "game_data.csv"
        _write_csv(
            csv_path,
            rows=[
                ("True", 2, 1),  # 2x Bolt, 1x Bear, won
                ("False", 0, 3),  # 0x Bolt, 3x Bear, lost
            ],
        )
        output_path = tmp_path / "out.jsonl"
        scanner = DeckOutcomeScanner(
            raw_csv_path=csv_path,
            card_binder=binder,
            output_path=output_path,
            source_game=GameId.MTG,
            expansion="MSH",
            format_code="PremierDraft",
        )

        result = scanner.scan()

        assert isinstance(result, DeckOutcomeScanResult)
        assert result.output_path == output_path
        assert result.unresolved_column_names == []

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

    def test_unresolved_card_excluded_from_deck_but_row_still_emitted(self, tmp_path: Path) -> None:
        binder = CardBinder()  # neither Bolt nor Bear registered
        csv_path = tmp_path / "game_data.csv"
        _write_csv(csv_path, rows=[("True", 1, 1)])
        output_path = tmp_path / "out.jsonl"
        scanner = DeckOutcomeScanner(
            raw_csv_path=csv_path,
            card_binder=binder,
            output_path=output_path,
            source_game=GameId.MTG,
            expansion="MSH",
            format_code="PremierDraft",
        )

        result = scanner.scan()

        assert sorted(result.unresolved_column_names) == ["Bear", "Bolt"]
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
        scanner = DeckOutcomeScanner(
            raw_csv_path=csv_path,
            card_binder=binder,
            output_path=output_path,
            source_game=GameId.MTG,
            expansion="MSH",
            format_code="PremierDraft",
        )

        scanner.scan()

        rows = _read_jsonl(output_path)
        assert len(rows) == 1
        assert rows[0]["deck_card_uuids"] == []


class TestDefaultOutputPath:
    def test_matches_default_output_dir_and_name(self) -> None:
        expected = DeckOutcomeScanner.DEFAULT_OUTPUT_DIR / "MSH.PremierDraft.deck_outcomes.jsonl"
        assert DeckOutcomeScanner.default_output_path("MSH", "PremierDraft") == expected

    def test_varies_by_expansion_and_format(self) -> None:
        assert DeckOutcomeScanner.default_output_path(
            "MSH", "PremierDraft"
        ) != DeckOutcomeScanner.default_output_path("MSH", "TradDraft")


class TestResume:
    def test_resuming_after_a_simulated_crash_produces_no_duplicates_or_gaps(
        self, tmp_path: Path
    ) -> None:
        binder = CardBinder()
        bolt = binder.add(_card("Bolt")).stored_card
        bear = binder.add(_card("Bear")).stored_card
        csv_path = tmp_path / "game_data.csv"
        rows = [("True", 1, 0), ("False", 0, 1), ("True", 2, 2), ("False", 1, 1)]
        _write_csv(csv_path, rows=rows)
        output_path = tmp_path / "out.jsonl"

        # Simulate a crash after 2 rows were durably written, followed
        # by a truncated (no closing brace) trailing partial line —
        # exactly what a process death mid-write() would leave behind.
        valid_prefix = "\n".join(
            json.dumps(
                {
                    "deck_card_uuids": [str(bolt.nocab_uuid)] * bolt_count
                    + [str(bear.nocab_uuid)] * bear_count,
                    "won": won == "True",
                    "expansion": "MSH",
                    "format": "PremierDraft",
                }
            )
            for won, bolt_count, bear_count in rows[:2]
        )
        output_path.write_text(valid_prefix + "\n" + '{"deck_card_uui')

        scanner = DeckOutcomeScanner(
            raw_csv_path=csv_path,
            card_binder=binder,
            output_path=output_path,
            source_game=GameId.MTG,
            expansion="MSH",
            format_code="PremierDraft",
        )
        scanner.scan()

        resumed_rows = _read_jsonl(output_path)
        assert len(resumed_rows) == len(rows)

        # A from-scratch scan over the same CSV should produce the same
        # multiset of rows (order isn't pinned by the plan — resume vs.
        # fresh may draw chunk boundaries differently in principle,
        # though not in this small a fixture).
        fresh_output_path = tmp_path / "fresh.jsonl"
        fresh_scanner = DeckOutcomeScanner(
            raw_csv_path=csv_path,
            card_binder=binder,
            output_path=fresh_output_path,
            source_game=GameId.MTG,
            expansion="MSH",
            format_code="PremierDraft",
        )
        fresh_scanner.scan()
        fresh_rows = _read_jsonl(fresh_output_path)

        def _sort_key(row: dict) -> tuple:
            return (tuple(sorted(row["deck_card_uuids"])), row["won"])

        assert sorted(resumed_rows, key=_sort_key) == sorted(fresh_rows, key=_sort_key)

    def test_fresh_scan_when_output_path_does_not_exist(self, tmp_path: Path) -> None:
        binder = CardBinder()
        binder.add(_card("Bolt"))
        binder.add(_card("Bear"))
        csv_path = tmp_path / "game_data.csv"
        _write_csv(csv_path, rows=[("True", 1, 0)])
        output_path = tmp_path / "does_not_exist_yet.jsonl"
        scanner = DeckOutcomeScanner(
            raw_csv_path=csv_path,
            card_binder=binder,
            output_path=output_path,
            source_game=GameId.MTG,
            expansion="MSH",
            format_code="PremierDraft",
        )

        scanner.scan()

        assert len(_read_jsonl(output_path)) == 1
