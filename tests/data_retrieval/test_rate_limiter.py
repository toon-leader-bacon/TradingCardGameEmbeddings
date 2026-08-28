from unittest.mock import patch

from src.data_retrieval.rate_limiter import RateLimiter


class TestRateLimiter:
    def test_first_call_does_not_sleep(self) -> None:
        limiter = RateLimiter(requests_per_minute=12)

        with patch("src.data_retrieval.rate_limiter.time.sleep") as mock_sleep:
            with patch(
                "src.data_retrieval.rate_limiter.time.monotonic", return_value=100.0
            ):
                limiter.wait()

        mock_sleep.assert_not_called()

    def test_second_call_within_window_sleeps_remaining_time(self) -> None:
        # requests_per_minute=12 -> minimum 5s between calls.
        limiter = RateLimiter(requests_per_minute=12)

        with patch("src.data_retrieval.rate_limiter.time.sleep") as mock_sleep:
            with patch(
                "src.data_retrieval.rate_limiter.time.monotonic",
                # 1st call: now=100.0 (no sleep). 2nd call: now=102.0
                # (needs to sleep 3s), then re-read post-sleep -> 105.0.
                side_effect=[100.0, 102.0, 105.0],
            ):
                limiter.wait()  # first call at t=100.0, no sleep
                limiter.wait()  # second call at t=102.0, 2s elapsed, needs 3s more

        mock_sleep.assert_called_once_with(3.0)

    def test_sleep_duration_counts_toward_next_calls_elapsed_time(self) -> None:
        # Regression test: a call's own sleep time must be reflected in
        # the timestamp used for the *next* call's elapsed-time check,
        # or paced calls can end up back-to-back with zero spacing.
        # requests_per_minute=12 -> minimum 5s between calls.
        limiter = RateLimiter(requests_per_minute=12)

        with patch("src.data_retrieval.rate_limiter.time.sleep") as mock_sleep:
            with patch(
                "src.data_retrieval.rate_limiter.time.monotonic",
                # call 1: now=0.0, no prior call -> no sleep, last=0.0
                # call 2: now=0.0 (immediate), elapsed=0 -> sleeps 5s,
                #   re-read post-sleep -> 5.0, last=5.0
                # call 3: now=5.0 (immediate), elapsed=5.0-5.0=0 -> sleeps
                #   5s again, re-read post-sleep -> 10.0
                side_effect=[0.0, 0.0, 5.0, 5.0, 10.0],
            ):
                limiter.wait()
                limiter.wait()
                limiter.wait()

        assert mock_sleep.call_args_list == [((5.0,),), ((5.0,),)]

    def test_call_after_full_interval_elapsed_does_not_sleep(self) -> None:
        limiter = RateLimiter(requests_per_minute=12)

        with patch("src.data_retrieval.rate_limiter.time.sleep") as mock_sleep:
            with patch(
                "src.data_retrieval.rate_limiter.time.monotonic",
                side_effect=[100.0, 106.0],
            ):
                limiter.wait()  # first call at t=100.0
                limiter.wait()  # second call at t=106.0, 6s elapsed >= 5s minimum

        mock_sleep.assert_not_called()
