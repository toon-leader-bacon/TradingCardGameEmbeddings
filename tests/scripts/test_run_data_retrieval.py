import importlib.util
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest

from src.data_retrieval.limitless_play.downloader import LimitlessGame

_SCRIPT = Path("scripts/run_data_retrieval.py")


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_data_retrieval_script", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_MODULE = _load_script()


class TestLimitlessGamesFrom:
    def test_none_is_empty(self) -> None:
        assert _MODULE.limitless_games_from(None) == ()

    def test_keeps_order_and_drops_repeats(self) -> None:
        assert _MODULE.limitless_games_from(["DCG", "OP", "DCG"]) == (
            LimitlessGame.DIGIMON,
            LimitlessGame.ONE_PIECE,
        )

    def test_all_is_every_game_in_declaration_order(self) -> None:
        assert _MODULE.limitless_games_from(["all"]) == tuple(LimitlessGame)

    def test_a_game_before_all_keeps_its_place(self) -> None:
        games = _MODULE.limitless_games_from(["SWU", "all"])
        assert games[0] == LimitlessGame.STAR_WARS_UNLIMITED
        assert sorted(games) == sorted(LimitlessGame)


class TestRunLimitlessPlay:
    def test_runs_each_game_in_series(self) -> None:
        with patch.object(_MODULE, "_run_standard") as run_standard:
            _MODULE.run_limitless_play((LimitlessGame.ONE_PIECE, LimitlessGame.DIGIMON))
        games = [call.args[0].game for call in run_standard.call_args_list]
        assert games == [LimitlessGame.ONE_PIECE, LimitlessGame.DIGIMON]

    def test_a_failed_game_does_not_stop_the_rest(self) -> None:
        with patch.object(
            _MODULE, "_run_standard", side_effect=[ValueError("none"), None]
        ) as run_standard:
            with pytest.raises(RuntimeError, match="OP"):
                _MODULE.run_limitless_play(
                    (LimitlessGame.ONE_PIECE, LimitlessGame.DIGIMON)
                )
        assert run_standard.call_count == 2
