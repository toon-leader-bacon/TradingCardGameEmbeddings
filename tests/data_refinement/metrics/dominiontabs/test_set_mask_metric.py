import json
from pathlib import Path

import pandas as pd
import pytest

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.card_binder.dominiontabs.ingestion_stage import (
    DominionTabsCardIngestionStage,
)
from src.data_refinement.metrics.dominiontabs.set_mask_metric import SetMaskMetric


def _write_raw(raw_dir: Path, db: list[dict], en: dict[str, dict]) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "cards_db.json").write_text(json.dumps(db))
    (raw_dir / "cards_en_us.json").write_text(json.dumps(en))


def _entry(card_tag: str, cardset_tags: list[str]) -> dict:
    return {
        "card_tag": card_tag,
        "cardset_tags": cardset_tags,
        "types": ["Action"],
        "cost": "3",
    }


def _scan(tmp_path: Path, db: list[dict], with_text: set[str]) -> pd.DataFrame:
    """Ingest via the real stage (only entries with English text become
    cards), then run the metric over the same raw directory."""
    en = {e["card_tag"]: {"description": "text", "name": e["card_tag"]} for e in db}
    en = {tag: entry for tag, entry in en.items() if tag in with_text}
    raw_dir = tmp_path / "raw"
    _write_raw(raw_dir, db, en)
    binder = CardBinder()
    DominionTabsCardIngestionStage().ingest(raw_dir, binder)
    output = tmp_path / "out" / "set_mask.parquet"
    metric = SetMaskMetric(binder, raw_data_dir=raw_dir, output_path=output)
    return pd.read_parquet(metric.scan())


class TestSetMaskMetricWithCardsMissingFromBinder:
    def test_one_entry_missing_from_a_large_binder_is_skipped_not_fatal(
        self, tmp_path: Path
    ) -> None:
        # 40 eligible entries; the stage skips one (no English text), which is
        # 2.5%, under the tolerated fraction.
        db = [_entry(f"Card {i}", ["allies"]) for i in range(40)]
        with_text = {e["card_tag"] for e in db} - {"Card 0"}

        rows = _scan(tmp_path, db, with_text)

        assert len(rows) == 39
        assert set(rows["label"]) == {"allies"}

    def test_many_entries_missing_from_the_binder_raise(self, tmp_path: Path) -> None:
        # A binder built from some other data would miss most raw entries.
        db = [_entry(f"Card {i}", ["allies"]) for i in range(10)]
        with_text = {"Card 0", "Card 1"}

        with pytest.raises(ValueError, match="same raw_data_dir"):
            _scan(tmp_path, db, with_text)

    def test_all_present_gives_one_row_per_eligible_entry(self, tmp_path: Path) -> None:
        db = [
            _entry("A", ["allies"]),
            _entry("B", ["nocturne"]),
            _entry("Copper", ["dominion1stEdition", "base"]),  # not one expansion
        ]

        rows = _scan(tmp_path, db, {"A", "B", "Copper"})

        assert sorted(rows["label"]) == ["allies", "nocturne"]
