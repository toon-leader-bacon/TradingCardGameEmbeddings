import importlib.util
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest

from src.schema.game_id import GameId

_SCRIPT = Path("scripts/run_card_binder_ingestion.py")


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "run_card_binder_ingestion_script", _SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_MODULE = _load_script()


def test_yugioh_has_one_card_source_ygoprodeck() -> None:
    yugioh_stages = [
        name
        for name, (stage, _) in _MODULE.STAGES.items()
        if stage.SOURCE_GAME is GameId.YUGIOH
    ]
    assert yugioh_stages == ["ygoprodeck_yugioh"]


def test_all_runs_every_stage_and_reports_a_failure() -> None:
    def run_one(name: str, raw_path: Path | None) -> None:
        if name == "scryfall":
            raise RuntimeError("boom")

    with patch.object(_MODULE, "run_one", side_effect=run_one) as mock_run:
        with pytest.raises(SystemExit):
            _MODULE.run_all()
    assert sorted(c.args[0] for c in mock_run.call_args_list) == sorted(_MODULE.STAGES)
