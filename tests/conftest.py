from unittest.mock import patch

import pytest


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
