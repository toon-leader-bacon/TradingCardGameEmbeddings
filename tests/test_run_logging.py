import argparse
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Iterator

import pytest

from src import run_logging
from src.run_logging import (
    RUN_LOG_NAME,
    IsoTimestampFormatter,
    LoggingSettings,
    LogLevel,
    add_logging_arguments,
    configure_logging,
    logging_settings_from,
    mirror_log_into,
    script_log_path,
    script_logger,
)

# Every logger configure_logging() reconfigures, from the module's own lists
_TOUCHED_LOGGER_NAMES = (
    *run_logging._PROJECT_LOGGER_NAMES,
    *run_logging._QUIET_LIBRARY_NAMES,
    *run_logging._SELF_HANDLED_LIBRARY_NAMES,
)


@pytest.fixture(autouse=True)
def _restored_logging_state() -> Iterator[None]:
    """configure_logging() rewires process-wide state (the root handlers,
    logger levels, sys.excepthook, warning capture). Snapshot it and put it
    back after each test, so pytest's own capture and other tests are
    unaffected.

    Inputs: none. Output: none (fixture).
    Side effects: restores logging state after the test; closes handlers
        the test added. Exceptions: none.
    """
    root = logging.getLogger()
    root_handlers, root_level = list(root.handlers), root.level
    named = {
        name: (
            list(logging.getLogger(name).handlers),
            logging.getLogger(name).level,
            logging.getLogger(name).propagate,
        )
        for name in _TOUCHED_LOGGER_NAMES
    }
    excepthook = sys.excepthook
    yield
    for handler in list(root.handlers):
        root.removeHandler(handler)
        if handler not in root_handlers:
            handler.close()
    for handler in root_handlers:
        root.addHandler(handler)
    root.setLevel(root_level)
    for name, (handlers, level, propagate) in named.items():
        library_logger = logging.getLogger(name)
        library_logger.handlers = handlers
        library_logger.setLevel(level)
        library_logger.propagate = propagate
    sys.excepthook = excepthook
    logging.captureWarnings(False)


def _flush_root_handlers() -> None:
    """Flush the root's handlers so a test can read their files.

    Inputs: none. Output: none. Side effects: flushes the root handlers.
    Exceptions: none.
    """
    for handler in logging.getLogger().handlers:
        handler.flush()


class TestIsoTimestampFormatter:
    def test_renders_milliseconds_and_a_utc_offset(self) -> None:
        record = logging.LogRecord("x", logging.INFO, "f.py", 1, "m", None, None)
        record.created = datetime(2026, 10, 11, 14, 3, 22, 417000).timestamp()
        stamp = IsoTimestampFormatter("%(asctime)s").format(record)
        assert stamp.startswith("2026-10-11T14:03:22.417")
        # ISO 8601 offset, e.g. -04:00 or +00:00
        assert stamp[-6] in "+-" and stamp[-3] == ":"


class TestScriptNames:
    def test_script_log_path_is_stem_then_timestamp(self) -> None:
        path = script_log_path(
            "scripts/run_metrics.py", datetime(2026, 10, 11, 14, 3, 22)
        )
        assert path.as_posix() == "logs/run_metrics/2026-10-11_140322.log"

    def test_script_log_path_honors_logs_root(self, tmp_path: Path) -> None:
        path = script_log_path("x.py", datetime(2026, 1, 2, 3, 4, 5), tmp_path)
        assert path == tmp_path / "x" / "2026-01-02_030405.log"

    def test_script_logger_is_named_under_scripts(self) -> None:
        assert script_logger("G:/repo/scripts/run_training.py").name == (
            "scripts.run_training"
        )


class TestArguments:
    def _parse(self, argv: list[str]) -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        add_logging_arguments(parser)
        return parser.parse_args(argv)

    def test_defaults_to_info_and_the_script_default(self, tmp_path: Path) -> None:
        settings = logging_settings_from(self._parse([]), tmp_path / "d.log")
        assert settings == LoggingSettings(LogLevel.INFO, tmp_path / "d.log")

    def test_flags_override_the_defaults(self, tmp_path: Path) -> None:
        args = self._parse(
            ["--log-file", str(tmp_path / "o.log"), "--log-level", "DEBUG"]
        )
        settings = logging_settings_from(args, tmp_path / "d.log")
        assert settings == LoggingSettings(LogLevel.DEBUG, tmp_path / "o.log")

    def test_none_default_means_console_only(self) -> None:
        assert logging_settings_from(self._parse([]), None).log_file is None

    def test_rejects_an_unknown_level(self) -> None:
        with pytest.raises(SystemExit):
            self._parse(["--log-level", "VERBOSE"])


class TestConfigureLogging:
    def test_file_gets_header_debug_and_source(self, tmp_path: Path) -> None:
        log_file = tmp_path / "nested" / "x.log"
        configure_logging(LoggingSettings(LogLevel.INFO, log_file))
        logging.getLogger("src.demo").debug("only in the file")
        _flush_root_handlers()

        lines = log_file.read_text(encoding="utf-8").splitlines()
        assert "started: argv=" in lines[0] and "commit=" in lines[0]
        assert "DEBUG   src.demo (test_run_logging.py:" in lines[1]
        assert lines[1].endswith("): only in the file")

    def test_console_filters_by_level_and_omits_dates(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        configure_logging(LoggingSettings(LogLevel.INFO, tmp_path / "x.log"))
        logging.getLogger("src.demo").debug("hidden")
        logging.getLogger("src.demo").info("shown")

        console = capsys.readouterr().err
        assert "hidden" not in console
        shown = [line for line in console.splitlines() if "shown" in line][0]
        assert shown.endswith(" INFO    src.demo: shown")
        assert shown[2] == ":" and "T" not in shown.split()[0]

    def test_library_debug_stays_out_of_the_file(self, tmp_path: Path) -> None:
        log_file = tmp_path / "x.log"
        configure_logging(LoggingSettings(LogLevel.DEBUG, log_file))
        logging.getLogger("some_library").debug("library chatter")
        logging.getLogger("urllib3").info("quiet library info")
        _flush_root_handlers()
        text = log_file.read_text(encoding="utf-8")
        assert "library chatter" not in text and "quiet library info" not in text

    def test_routes_self_handled_libraries_through_root(self, tmp_path: Path) -> None:
        library_logger = logging.getLogger("transformers")
        own_handler = logging.StreamHandler()
        library_logger.addHandler(own_handler)
        library_logger.propagate = False

        log_file = tmp_path / "x.log"
        configure_logging(LoggingSettings(LogLevel.INFO, log_file))
        library_logger.warning("from transformers")
        _flush_root_handlers()

        assert own_handler not in library_logger.handlers
        assert library_logger.propagate
        assert "from transformers" in log_file.read_text(encoding="utf-8")

    def test_captures_warnings(self, tmp_path: Path) -> None:
        log_file = tmp_path / "x.log"
        configure_logging(LoggingSettings(LogLevel.INFO, log_file))
        warnings.warn("a captured warning", UserWarning, stacklevel=1)
        _flush_root_handlers()
        assert "a captured warning" in log_file.read_text(encoding="utf-8")

    def test_logs_uncaught_exceptions(self, tmp_path: Path) -> None:
        log_file = tmp_path / "x.log"
        configure_logging(LoggingSettings(LogLevel.INFO, log_file))
        try:
            raise RuntimeError("boom")
        except RuntimeError as error:
            sys.excepthook(RuntimeError, error, error.__traceback__)
        _flush_root_handlers()
        text = log_file.read_text(encoding="utf-8")
        assert "CRITICAL" in text and "RuntimeError: boom" in text

    def test_a_second_call_starts_over(self, tmp_path: Path) -> None:
        configure_logging(LoggingSettings(LogLevel.INFO, tmp_path / "a.log"))
        configure_logging(LoggingSettings(LogLevel.INFO, None))
        handlers = logging.getLogger().handlers
        assert len(handlers) == 1
        assert not isinstance(handlers[0], logging.FileHandler)


class TestMirrorLogInto:
    def test_run_log_holds_everything_from_the_first_line(self, tmp_path: Path) -> None:
        script_log = tmp_path / "logs" / "x.log"
        configure_logging(LoggingSettings(LogLevel.INFO, script_log))
        logging.getLogger("src.demo").info("before the run directory")
        run_directory = tmp_path / "run"
        run_directory.mkdir()

        mirror_log_into(run_directory / RUN_LOG_NAME)
        logging.getLogger("src.demo").info("after the run directory")
        _flush_root_handlers()

        run_log = (run_directory / RUN_LOG_NAME).read_text(encoding="utf-8")
        assert run_log == script_log.read_text(encoding="utf-8")
        assert "before the run directory" in run_log
        assert "after the run directory" in run_log

    def test_mirroring_into_the_current_file_is_a_no_op(self, tmp_path: Path) -> None:
        log_file = tmp_path / RUN_LOG_NAME
        configure_logging(LoggingSettings(LogLevel.INFO, log_file))
        mirror_log_into(log_file)
        assert len(logging.getLogger().handlers) == 2  # console + the one file

    def test_rejects_console_only_logging(self, tmp_path: Path) -> None:
        configure_logging(LoggingSettings(LogLevel.INFO, None))
        with pytest.raises(ValueError, match="no log file"):
            mirror_log_into(tmp_path / RUN_LOG_NAME)
