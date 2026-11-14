import importlib.util
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest

from src.data_retrieval.limitless_play.downloader import LimitlessGame
from src.data_retrieval.mtgtop8.formats import MtgTop8Format

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


class TestMtgTop8FormatsFrom:
    def test_none_is_every_format_in_declaration_order(self) -> None:
        assert _MODULE.mtgtop8_formats_from(None) == tuple(MtgTop8Format)

    def test_keeps_order_and_drops_repeats(self) -> None:
        assert _MODULE.mtgtop8_formats_from(["LE", "VI", "LE"]) == (
            MtgTop8Format.LEGACY,
            MtgTop8Format.VINTAGE,
        )


class TestRunMtgTop8:
    def test_runs_formats_in_series_on_one_rate_limiter(self) -> None:
        with patch.object(_MODULE, "_run_standard") as run_standard:
            _MODULE.run_mtgtop8(
                (MtgTop8Format.VINTAGE, MtgTop8Format.LEGACY),
                _MODULE.CrawlSessionOptions(requests_per_minute=30, max_items=5),
            )
        downloaders = [call.args[0] for call in run_standard.call_args_list]
        assert [d.deck_format for d in downloaders] == [
            MtgTop8Format.VINTAGE,
            MtgTop8Format.LEGACY,
        ]
        assert downloaders[0].rate_limiter is downloaders[1].rate_limiter
        assert downloaders[0].rate_limiter.requests_per_minute == 30
        assert downloaders[0].max_decks_per_session == 5

    def test_defaults_to_robots_txt_pace(self) -> None:
        with patch.object(_MODULE, "_run_standard") as run_standard:
            _MODULE.run_mtgtop8((MtgTop8Format.PAUPER,))
        assert run_standard.call_args.args[0].rate_limiter.requests_per_minute == 12

    def test_a_failed_format_stops_the_rest(self) -> None:
        with patch.object(
            _MODULE, "_run_standard", side_effect=ValueError("blocked")
        ) as run_standard:
            with pytest.raises(ValueError, match="blocked"):
                _MODULE.run_mtgtop8((MtgTop8Format.VINTAGE, MtgTop8Format.LEGACY))
        assert run_standard.call_count == 1
