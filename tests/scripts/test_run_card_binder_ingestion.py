import importlib.util
from pathlib import Path
from types import ModuleType
from unittest.mock import call, patch

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


class TestFollowUpStages:
    def test_every_follow_up_is_a_stage(self) -> None:
        for name, follow_up in _MODULE.FOLLOW_UP_STAGES.items():
            assert name in _MODULE.STAGES
            assert follow_up.name in _MODULE.STAGES

    def test_tcgjson_yugioh_runs_the_passcode_aliases_after_it(self) -> None:
        follow_up = _MODULE.FollowUpStage("ygoprodeck_yugioh_passcodes", lambda: True)
        with (
            patch.object(_MODULE, "run_one") as run_one,
            patch.dict(_MODULE.FOLLOW_UP_STAGES, {"tcgjson_yugioh": follow_up}),
        ):
            _MODULE.run_with_follow_up("tcgjson_yugioh", raw_path=None)
        assert run_one.call_args_list == [
            call("tcgjson_yugioh", None),
            call("ygoprodeck_yugioh_passcodes", raw_path=None),
        ]

    def test_a_missing_follow_up_input_is_skipped(self) -> None:
        follow_up = _MODULE.FollowUpStage("ygoprodeck_yugioh_passcodes", lambda: False)
        with (
            patch.object(_MODULE, "run_one") as run_one,
            patch.dict(_MODULE.FOLLOW_UP_STAGES, {"tcgjson_yugioh": follow_up}),
        ):
            _MODULE.run_with_follow_up("tcgjson_yugioh", raw_path=None)
        assert run_one.call_args_list == [call("tcgjson_yugioh", None)]

    def test_all_runs_a_follow_up_only_after_its_stage(self) -> None:
        with patch.object(_MODULE, "run_with_follow_up") as run_with_follow_up:
            _MODULE.run_all()
        names = [c.args[0] for c in run_with_follow_up.call_args_list]
        assert "ygoprodeck_yugioh_passcodes" not in names
        assert "tcgjson_yugioh" in names

    def test_a_failed_follow_up_is_reported_under_its_own_name(self) -> None:
        follow_up = _MODULE.FollowUpStage("ygoprodeck_yugioh_passcodes", lambda: True)

        def _run_one(name: str, raw_path: object = None) -> None:
            if name == "ygoprodeck_yugioh_passcodes":
                raise FileNotFoundError("no mirror")

        with (
            patch.object(_MODULE, "run_one", side_effect=_run_one),
            patch.dict(_MODULE.FOLLOW_UP_STAGES, {"tcgjson_yugioh": follow_up}),
        ):
            try:
                _MODULE.run_with_follow_up("tcgjson_yugioh", raw_path=None)
            except _MODULE.FollowUpStageError as error:
                assert _MODULE._failed_stage_name("tcgjson_yugioh", error) == (
                    "ygoprodeck_yugioh_passcodes"
                )
            else:
                raise AssertionError("expected FollowUpStageError")
