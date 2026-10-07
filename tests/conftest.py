from pathlib import Path
from unittest.mock import patch

import pytest

from src.dojos.generic.dojo_config import SPLIT_DIRECTORY_ENV_VAR


@pytest.fixture(autouse=True)
def _no_real_sleep():
    """Every retry/backoff path in this project calls time.sleep(), and
    several downloader tests exercise those paths for real (failure,
    retry-then-succeed, retries-exhausted) - without this, the suite spends
    most of its wall-clock time actually asleep rather than testing
    anything. A test that needs to assert on sleep calls/durations still
    patches time.sleep itself; that local patch just supersedes this one
    for its own duration.

    Inputs: none.
    Output: none (fixture).
    Side effects: patches time.sleep to a no-op for the duration of each
        test.
    Exceptions: none.
    """
    with patch("time.sleep"):
        yield


@pytest.fixture(autouse=True)
def _isolated_split_directory(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Point DojoConfig's default split directory at a fresh temp
    directory, so a test that builds a dojo without an output_directory
    never writes into (or reuses files from) the real data/splits/.

    Inputs: none.
    Output: the temp directory.
    Side effects: sets $NOCAB_SPLIT_DIRECTORY for one test.
    Exceptions: none.
    """
    directory = tmp_path_factory.mktemp("splits")
    monkeypatch.setenv(SPLIT_DIRECTORY_ENV_VAR, str(directory))
    return directory
