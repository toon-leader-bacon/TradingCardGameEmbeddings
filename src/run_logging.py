"""Project-wide logging setup for scripts: one call at the top of main().

Every script calls configure_logging() once, in place of
logging.basicConfig(). Records then go to two places:

    console   HH:MM:SS LEVEL logger.name: message
              level from --log-level (default INFO); written through
              tqdm.write so lines don't tear progress bars
    file      2026-10-11T14:03:22.417-04:00 LEVEL logger.name (file.py:123): message
              everything at DEBUG and above from src.* and scripts.*,
              INFO and above from other libraries (WARNING for the
              chatty ones in _QUIET_LIBRARY_NAMES)

--log-level DEBUG lowers the console to DEBUG for src.* and scripts.*
only; library DEBUG records stay off everywhere.

Warnings (warnings.warn) and uncaught exceptions are logged too, and the
file opens with a header line recording what produced it (argv, git
commit, host, Python and torch versions).

Library code logs through a module logger (logging.getLogger(__name__)),
never print() or tqdm.write(). A script may still print() what it
produces (a --list listing, a CSV or report on stdout), so stdout stays
pipeable; what the program says about itself goes through the logger.

Typical script:

    _logger = script_logger(__file__)

    def main() -> int:
        parser = argparse.ArgumentParser(...)
        add_logging_arguments(parser)
        args = parser.parse_args()
        configure_logging(logging_settings_from(args, script_log_path(__file__)))
        _logger.info("starting")
"""

import argparse
import logging
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from enum import IntEnum
from pathlib import Path
from types import TracebackType

from tqdm import tqdm

# Default home for logs that belong to no run directory
LOGS_ROOT = Path("logs")

# A run directory's log file (training, evaluation)
RUN_LOG_NAME = "run.log"

# Loggers whose DEBUG records reach the file; everything else stops at INFO
_PROJECT_LOGGER_NAMES = ("src", "scripts")

# Third-party loggers too chatty at INFO; held at WARNING
_QUIET_LIBRARY_NAMES = (
    "urllib3",
    "filelock",
    "fsspec",
    "httpx",
    "huggingface_hub",
    "matplotlib",
    "PIL",
    "transformers",
)

# Libraries that log through their own stderr handler (transformers also
# stops propagation), bypassing the root handlers: tears tqdm bars and
# misses the file. Their handlers are removed and propagation restored.
# This assumes they're imported before configure_logging() runs (true while
# every import is at module level); a later first import re-attaches them.
_SELF_HANDLED_LIBRARY_NAMES = ("transformers", "huggingface_hub")

_CONSOLE_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_CONSOLE_DATE_FORMAT = "%H:%M:%S"
_FILE_FORMAT = (
    "%(asctime)s %(levelname)-7s %(name)s (%(filename)s:%(lineno)d): %(message)s"
)


class LogLevel(IntEnum):
    """The console levels a script accepts, valued as logging's own ints."""

    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR


@dataclass(frozen=True)
class LoggingSettings:
    """How one script invocation logs.

    console_level: the lowest level shown on the console.
    log_file: the file every record is appended to, or None for
        console-only logging (report scripts, whose printed output is the
        product and needs no log file).
    """

    console_level: LogLevel
    log_file: Path | None


class IsoTimestampFormatter(logging.Formatter):
    """A Formatter whose %(asctime)s is ISO 8601 local time with
    milliseconds and UTC offset, e.g. 2026-10-11T14:03:22.417-04:00."""

    def formatTime(  # noqa: N802 (overrides logging.Formatter's name)
        self, record: logging.LogRecord, datefmt: str | None = None
    ) -> str:
        """Render record.created as ISO 8601 local time, ignoring datefmt.

        Inputs: record (LogRecord); datefmt (str | None, unused: the
            format is fixed). Output: str.
        Side effects: none. Exceptions: none.
        """
        moment = datetime.fromtimestamp(record.created).astimezone()
        return moment.isoformat(timespec="milliseconds")


class TqdmConsoleHandler(logging.Handler):
    """Writes formatted records to stderr through tqdm.write, so a log line
    printed while a tqdm bar is live lands above the bar, not through it."""

    def emit(self, record: logging.LogRecord) -> None:
        """Format record and write it, plus a newline, to stderr via tqdm.

        Inputs: record (LogRecord). Output: none.
        Side effects: writes one line to stderr.
        Exceptions: none; a failure goes to logging's handleError().
        """
        try:
            # sys.stderr looked up per record: tests and tools may swap it
            tqdm.write(self.format(record), file=sys.stderr)
        except Exception:  # logging's contract: never raise from emit()
            self.handleError(record)


def configure_logging(settings: LoggingSettings) -> None:
    """Replace the root logger's handlers with this project's console and
    (optionally) file handlers, and route warnings and uncaught exceptions
    through logging. Safe to call more than once: each call starts over.

    Inputs: settings (LoggingSettings).
    Output: none.
    Side effects: removes and closes every existing root handler; sets the
        root, project and quiet-library logger levels; strips the
        self-handled libraries' own handlers and restores their
        propagation (they configure themselves on import, which in a
        script happens at module level, before main() calls this);
        creates settings.log_file's parent directories and opens the file
        for appending; enables logging.captureWarnings; replaces
        sys.excepthook; logs one startup header line.
    Exceptions: OSError if the log file can't be created or opened.

    Example:
        >>> configure_logging(LoggingSettings(LogLevel.INFO, Path("logs/x/run.log")))
    """
    root = logging.getLogger()

    # Start from a clean root: drop basicConfig's or a previous call's handlers
    _remove_root_handlers(root)

    # Levels: the root passes INFO; project loggers pass DEBUG on to the
    # handlers, which then filter per destination
    root.setLevel(logging.INFO)
    for name in _PROJECT_LOGGER_NAMES:
        logging.getLogger(name).setLevel(logging.DEBUG)
    for name in _QUIET_LIBRARY_NAMES:
        logging.getLogger(name).setLevel(logging.WARNING)

    # Pull libraries with their own stderr handlers back under ours
    for name in _SELF_HANDLED_LIBRARY_NAMES:
        _route_library_through_root(logging.getLogger(name))

    # Attach the destinations
    root.addHandler(_console_handler(settings.console_level))
    if settings.log_file is not None:
        root.addHandler(_file_handler(settings.log_file))

    # Route the remaining channels through logging, then say who we are
    logging.captureWarnings(True)
    sys.excepthook = _log_uncaught_exception
    logging.getLogger(__name__).info("%s", _startup_header())


def mirror_log_into(path: Path) -> None:
    """Copy the current log file's contents so far into path, then keep
    logging into path as well. For a run directory that only exists after
    logging has started (run_training.py), so its run.log still holds the
    whole run from the first line.

    Inputs: path (Path), the new log file; its parent must exist (the
        run directory, already created by the caller).
    Output: none.
    Side effects: flushes the current file handler; writes path with the
        current log file's contents; adds a second file handler on the
        root logger appending to path. None if path already is the
        current log file (--log-file pointed there).
    Exceptions: ValueError if configure_logging() set up no log file
        (nothing to mirror); OSError if path can't be written.

    Example:
        >>> mirror_log_into(config.run_directory / RUN_LOG_NAME)
    """
    root = logging.getLogger()

    # Find the log file configure_logging() opened
    current = _current_file_handler(root)
    if current is None:
        raise ValueError("no log file is configured; nothing to mirror")
    current.flush()
    if path.resolve() == Path(current.baseFilename).resolve():
        return

    # Seed the new file with everything so far, then log into it too
    _copy_log_contents(Path(current.baseFilename), path)
    root.addHandler(_file_handler(path))


def add_logging_arguments(parser: argparse.ArgumentParser) -> None:
    """Add --log-file and --log-level to a script's parser.

    Inputs: parser (ArgumentParser).
    Output: none.
    Side effects: adds two optional arguments to parser: --log-file (Path,
        default None: the script's own default) and --log-level (one of
        LogLevel's names, default INFO).
    Exceptions: argparse.ArgumentError if parser already defines either.

    Example:
        >>> add_logging_arguments(parser)
    """
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="append the full log here (default: the script's own log file)",
    )
    parser.add_argument(
        "--log-level",
        choices=[level.name for level in LogLevel],
        default=LogLevel.INFO.name,
        help="lowest level shown on the console (the file always gets DEBUG)",
    )


def logging_settings_from(
    args: argparse.Namespace, default_log_file: Path | None
) -> LoggingSettings:
    """Parse the arguments add_logging_arguments() added into
    LoggingSettings, falling back to the script's default log file.

    Inputs: args (Namespace from a parser add_logging_arguments() saw);
        default_log_file (Path | None), used when --log-file is absent;
        None means console only.
    Output: LoggingSettings.
    Side effects: none.
    Exceptions: AttributeError if args lacks log_file or log_level.

    Example:
        >>> logging_settings_from(args, script_log_path(__file__))
    """
    log_file = args.log_file if args.log_file is not None else default_log_file
    return LoggingSettings(LogLevel[args.log_level], log_file)


def script_log_path(
    script_file: str, started_at: datetime | None = None, logs_root: Path = LOGS_ROOT
) -> Path:
    """The default log file for one invocation of a script:
    logs_root/<script stem>/<YYYY-MM-DD_HHMMSS>.log.

    Inputs: script_file (str), the script's __file__; started_at
        (datetime | None), default now; logs_root (Path).
    Output: Path (not created).
    Side effects: none (reads the clock when started_at is None).
    Exceptions: none.

    Example:
        >>> script_log_path("scripts/run_metrics.py", datetime(2026, 10, 11, 14, 3, 22)).as_posix()
        'logs/run_metrics/2026-10-11_140322.log'
    """
    moment = started_at if started_at is not None else datetime.now()
    return logs_root / Path(script_file).stem / f"{moment:%Y-%m-%d_%H%M%S}.log"


def script_logger(script_file: str) -> logging.Logger:
    """The logger for a script, named scripts.<stem> rather than __main__,
    so its lines say which script wrote them and it sits under the
    project's DEBUG-to-file "scripts" logger.

    Inputs: script_file (str), the script's __file__.
    Output: logging.Logger.
    Side effects: none (logging caches the logger).
    Exceptions: none.

    Example:
        >>> _logger = script_logger(__file__)
    """
    return logging.getLogger(f"scripts.{Path(script_file).stem}")


def _remove_root_handlers(root: logging.Logger) -> None:
    """Remove and close every handler on root.

    Inputs: root (Logger). Output: none.
    Side effects: closes the handlers' streams/files. Exceptions: none.
    """
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()


def _route_library_through_root(library_logger: logging.Logger) -> None:
    """Remove and close library_logger's own handlers and set it to
    propagate, so its records reach the root's console and file handlers.

    Inputs: library_logger (Logger). Output: none.
    Side effects: mutates library_logger's handlers and propagate flag.
    Exceptions: none.
    """
    for handler in list(library_logger.handlers):
        library_logger.removeHandler(handler)
        handler.close()
    library_logger.propagate = True


def _console_handler(level: LogLevel) -> TqdmConsoleHandler:
    """A TqdmConsoleHandler at level with the console format.

    Inputs: level (LogLevel). Output: TqdmConsoleHandler.
    Side effects: none. Exceptions: none.
    """
    handler = TqdmConsoleHandler(level)
    handler.setFormatter(logging.Formatter(_CONSOLE_FORMAT, _CONSOLE_DATE_FORMAT))
    return handler


def _file_handler(path: Path) -> logging.FileHandler:
    """A UTF-8, append-mode FileHandler at DEBUG with the file format,
    creating path's parent directories first.

    Inputs: path (Path). Output: logging.FileHandler.
    Side effects: creates directories; opens path for appending.
    Exceptions: OSError if either fails.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, mode="a", encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(IsoTimestampFormatter(_FILE_FORMAT))
    return handler


def _current_file_handler(root: logging.Logger) -> logging.FileHandler | None:
    """The first FileHandler on root (the one configure_logging() opened),
    or None if logging is console only.

    Inputs: root (Logger). Output: FileHandler | None.
    Side effects: none. Exceptions: none.
    """
    for handler in root.handlers:
        if isinstance(handler, logging.FileHandler):
            return handler
    return None


def _copy_log_contents(source: Path, destination: Path) -> None:
    """Write source's bytes to destination, replacing it.

    Inputs: source, destination (Path). Output: none.
    Side effects: writes destination. Exceptions: OSError.
    """
    shutil.copyfile(source, destination)


def _startup_header() -> str:
    """One line naming what produced this log: argv, git commit (or
    "unknown" outside a repo or without git), hostname, Python version,
    and torch version only if torch is already imported (never imports
    it: that costs seconds for scripts that don't use it).

    Inputs: none. Output: str.
    Side effects: runs `git rev-parse --short HEAD` once.
    Exceptions: none; a failed git call yields "unknown".
    """
    torch = sys.modules.get("torch")
    torch_version = getattr(torch, "__version__", "not loaded")
    return (
        f"started: argv={sys.argv} commit={_git_commit()} "
        f"host={platform.node()} python={platform.python_version()} "
        f"torch={torch_version}"
    )


def _log_uncaught_exception(
    exc_type: type[BaseException],
    exc_value: BaseException,
    exc_traceback: TracebackType | None,
) -> None:
    """sys.excepthook: log an uncaught exception's traceback at CRITICAL,
    except KeyboardInterrupt, which goes to the default hook unlogged.
    Covers the main thread only (sys.excepthook, not threading.excepthook):
    no script runs work in threads; DataLoader workers are processes.

    Inputs: the excepthook triple. Output: none.
    Side effects: one CRITICAL record with the traceback (reaching the
        console and the file). Exceptions: none.
    """
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logging.getLogger(__name__).critical(
        "uncaught exception", exc_info=(exc_type, exc_value, exc_traceback)
    )


def _git_commit() -> str:
    """The short hash of HEAD in the repo holding this file, or "unknown"
    without git or outside a repo.

    Inputs: none. Output: str.
    Side effects: runs `git rev-parse --short HEAD`.
    Exceptions: none.
    """
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return completed.stdout.strip() or "unknown"
