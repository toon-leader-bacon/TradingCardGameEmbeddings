from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.data_retrieval.rate_limiter import RateLimiter
from src.data_retrieval.download_utils import (
    append_with_manifest,
    download_to_file,
    download_to_string,
    paced_get,
    read_manifest,
)


def _mock_streaming_response(chunks: list[bytes]) -> MagicMock:
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.iter_content.return_value = iter(chunks)
    response.__enter__.return_value = response
    response.__exit__.return_value = False
    return response


def _mock_text_response(text: str) -> MagicMock:
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.text = text
    return response


class TestDownloadToFile:
    def test_writes_streamed_response_to_destination_path(self, tmp_path: Path) -> None:
        destination_path = tmp_path / "file.bin"
        response = _mock_streaming_response([b"chunk-one-", b"chunk-two"])

        with patch("requests.get", return_value=response) as mock_get:
            download_to_file("https://example.com/file.bin", destination_path)

        mock_get.assert_called_once()
        assert mock_get.call_args.args[0] == "https://example.com/file.bin"
        assert mock_get.call_args.kwargs["stream"] is True
        assert destination_path.read_bytes() == b"chunk-one-chunk-two"

    def test_creates_parent_directory_if_missing(self, tmp_path: Path) -> None:
        destination_path = tmp_path / "does" / "not" / "exist" / "file.bin"
        response = _mock_streaming_response([b"data"])

        with patch("requests.get", return_value=response):
            download_to_file("https://example.com/file.bin", destination_path)

        assert destination_path.exists()

    def test_uses_given_timeout_and_chunk_size(self, tmp_path: Path) -> None:
        destination_path = tmp_path / "file.bin"
        response = _mock_streaming_response([b"data"])

        with patch("requests.get", return_value=response) as mock_get:
            download_to_file(
                "https://example.com/file.bin",
                destination_path,
                timeout_seconds=5,
                chunk_size_bytes=64,
            )

        assert mock_get.call_args.kwargs["timeout"] == 5
        response.iter_content.assert_called_once_with(chunk_size=64)

    def test_raises_on_http_error(self, tmp_path: Path) -> None:
        destination_path = tmp_path / "file.bin"
        response = _mock_streaming_response([])
        response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")

        with patch("requests.get", return_value=response):
            with pytest.raises(requests.HTTPError):
                download_to_file("https://example.com/file.bin", destination_path)

    def test_sends_given_headers(self, tmp_path: Path) -> None:
        destination_path = tmp_path / "file.bin"
        response = _mock_streaming_response([b"data"])

        with patch("requests.get", return_value=response) as mock_get:
            download_to_file(
                "https://example.com/file.bin",
                destination_path,
                headers={"User-Agent": "test-agent"},
            )

        assert mock_get.call_args.kwargs["headers"] == {"User-Agent": "test-agent"}

    def test_omitting_headers_sends_none(self, tmp_path: Path) -> None:
        destination_path = tmp_path / "file.bin"
        response = _mock_streaming_response([b"data"])

        with patch("requests.get", return_value=response) as mock_get:
            download_to_file("https://example.com/file.bin", destination_path)

        assert mock_get.call_args.kwargs["headers"] is None

    def test_removes_partial_file_on_write_failure(self, tmp_path: Path) -> None:
        destination_path = tmp_path / "file.bin"
        response = _mock_streaming_response([b"first-chunk"])
        response.iter_content.side_effect = requests.ConnectionError("dropped")

        with patch("requests.get", return_value=response):
            with pytest.raises(requests.ConnectionError):
                download_to_file("https://example.com/file.bin", destination_path)

        assert not destination_path.exists()


class TestDownloadToString:
    def test_returns_response_text(self) -> None:
        response = _mock_text_response('{"hello": "world"}')

        with patch("requests.get", return_value=response) as mock_get:
            body = download_to_string("https://example.com/page.json")

        mock_get.assert_called_once_with(
            "https://example.com/page.json", timeout=60, headers=None
        )
        assert body == '{"hello": "world"}'

    def test_uses_given_timeout(self) -> None:
        response = _mock_text_response("data")

        with patch("requests.get", return_value=response) as mock_get:
            download_to_string("https://example.com/page.json", timeout_seconds=5)

        assert mock_get.call_args.kwargs["timeout"] == 5

    def test_sends_given_headers(self) -> None:
        response = _mock_text_response("data")

        with patch("requests.get", return_value=response) as mock_get:
            download_to_string(
                "https://example.com/page.json", headers={"User-Agent": "test-agent"}
            )

        assert mock_get.call_args.kwargs["headers"] == {"User-Agent": "test-agent"}

    def test_omitting_headers_sends_none(self) -> None:
        response = _mock_text_response("data")

        with patch("requests.get", return_value=response) as mock_get:
            download_to_string("https://example.com/page.json")

        assert mock_get.call_args.kwargs["headers"] is None

    def test_retries_then_succeeds(self) -> None:
        failing_response = _mock_text_response("")
        failing_response.raise_for_status.side_effect = requests.ConnectionError(
            "dropped"
        )
        succeeding_response = _mock_text_response("data")

        with patch(
            "requests.get", side_effect=[failing_response, succeeding_response]
        ) as mock_get:
            body = download_to_string("https://example.com/page.json")

        assert mock_get.call_count == 2
        assert body == "data"

    def test_raises_after_max_attempts_exhausted(self) -> None:
        response = _mock_text_response("")
        response.raise_for_status.side_effect = requests.HTTPError("500 Server Error")

        with patch("requests.get", return_value=response) as mock_get:
            with pytest.raises(requests.HTTPError):
                download_to_string("https://example.com/page.json", max_attempts=2)

        assert mock_get.call_count == 2


class TestReadManifest:
    def test_returns_empty_set_when_manifest_does_not_exist(
        self, tmp_path: Path
    ) -> None:
        assert read_manifest(tmp_path / "manifest.txt") == set()

    def test_returns_recorded_ids(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / "manifest.txt"
        manifest_path.write_text("a\nb\nc\n", encoding="utf-8")

        assert read_manifest(manifest_path) == {"a", "b", "c"}

    def test_skips_blank_lines(self, tmp_path: Path) -> None:
        manifest_path = tmp_path / "manifest.txt"
        manifest_path.write_text("a\n\nb\n\n", encoding="utf-8")

        assert read_manifest(manifest_path) == {"a", "b"}


class TestAppendWithManifest:
    def test_appends_data_line_and_manifest_id(self, tmp_path: Path) -> None:
        data_path = tmp_path / "data.jsonl"
        manifest_path = tmp_path / "manifest.txt"

        append_with_manifest(data_path, manifest_path, '{"id": "a"}', "a")
        append_with_manifest(data_path, manifest_path, '{"id": "b"}', "b")

        assert data_path.read_text(encoding="utf-8") == '{"id": "a"}\n{"id": "b"}\n'
        assert manifest_path.read_text(encoding="utf-8") == "a\nb\n"

    def test_writes_data_line_before_manifest_line(self, tmp_path: Path) -> None:
        # Regression guard for the crash-safety write ordering this
        # function exists to centralize: a crash between the two
        # writes must leave a data row with no matching manifest
        # entry (harmless, re-fetchable), never the reverse.
        data_path = tmp_path / "data.jsonl"
        manifest_path = tmp_path / "manifest.txt"

        real_open = open
        write_order: list[str] = []

        def _tracking_open(path, mode="r", *args, **kwargs):
            if mode == "a":
                write_order.append(Path(path).name)
            return real_open(path, mode, *args, **kwargs)

        with patch(
            "src.data_retrieval.download_utils.open",
            side_effect=_tracking_open,
            create=True,
        ):
            append_with_manifest(data_path, manifest_path, "data", "id")

        assert write_order == ["data.jsonl", "manifest.txt"]


def _status_response(status: int, headers: dict[str, str] | None = None) -> MagicMock:
    response = MagicMock()
    response.status_code = status
    response.headers = headers or {}
    if status >= 400:
        response.raise_for_status.side_effect = requests.HTTPError(f"HTTP {status}")
    else:
        response.raise_for_status.return_value = None
    return response


def _unlimited() -> RateLimiter:
    # Effectively unlimited, so wait() never sleeps in tests
    return RateLimiter(requests_per_minute=1_000_000_000)


class TestPacedGet:
    def test_returns_a_2xx_response(self) -> None:
        response = _status_response(200)
        with patch("requests.get", return_value=response):
            assert paced_get("https://x.test", rate_limiter=_unlimited()) is response

    def test_returns_a_tolerated_status_without_retrying(self) -> None:
        with patch("requests.get", return_value=_status_response(404)) as get:
            response = paced_get(
                "https://x.test",
                rate_limiter=_unlimited(),
                tolerated_statuses=frozenset({404}),
            )
        assert response.status_code == 404
        get.assert_called_once()

    def test_an_untolerated_status_raises_after_retries(self) -> None:
        with (
            patch("requests.get", return_value=_status_response(404)) as get,
            patch("src.data_retrieval.download_utils.time.sleep"),
            pytest.raises(requests.HTTPError),
        ):
            paced_get("https://x.test", rate_limiter=_unlimited(), max_attempts=2)
        assert get.call_count == 2

    def test_a_429_is_waited_out_by_retry_after(self) -> None:
        responses = [_status_response(429, {"Retry-After": "7"}), _status_response(200)]
        with (
            patch("requests.get", side_effect=responses),
            patch("src.data_retrieval.download_utils.time.sleep") as sleep,
        ):
            response = paced_get("https://x.test", rate_limiter=_unlimited())
        assert response.status_code == 200
        sleep.assert_called_once_with(7.0)

    def test_a_429_without_retry_after_waits_a_minute(self) -> None:
        responses = [_status_response(429), _status_response(200)]
        with (
            patch("requests.get", side_effect=responses),
            patch("src.data_retrieval.download_utils.time.sleep") as sleep,
        ):
            paced_get("https://x.test", rate_limiter=_unlimited())
        sleep.assert_called_once_with(60.0)

    def test_still_rate_limited_after_every_wait_raises(self) -> None:
        with (
            patch("requests.get", return_value=_status_response(429)) as get,
            patch("src.data_retrieval.download_utils.time.sleep"),
            pytest.raises(requests.HTTPError, match="still rate limited"),
        ):
            paced_get(
                "https://x.test", rate_limiter=_unlimited(), max_rate_limited_waits=2
            )
        assert get.call_count == 3

    def test_a_huge_retry_after_is_capped(self) -> None:
        responses = [
            _status_response(429, {"Retry-After": "86400"}),
            _status_response(200),
        ]
        with (
            patch("requests.get", side_effect=responses),
            patch("src.data_retrieval.download_utils.time.sleep") as sleep,
        ):
            paced_get("https://x.test", rate_limiter=_unlimited())
        sleep.assert_called_once_with(900.0)

    def test_negative_rate_limited_waits_raise(self) -> None:
        with pytest.raises(ValueError, match="max_rate_limited_waits"):
            paced_get(
                "https://x.test", rate_limiter=_unlimited(), max_rate_limited_waits=-1
            )

    def test_tolerating_429_raises(self) -> None:
        with pytest.raises(ValueError, match="429"):
            paced_get(
                "https://x.test",
                rate_limiter=_unlimited(),
                tolerated_statuses=frozenset({429}),
            )
