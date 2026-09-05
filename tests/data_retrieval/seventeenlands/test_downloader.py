import gzip
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.data_retrieval.rate_limiter import RateLimiter
from src.data_retrieval.seventeenlands.downloader import (
    DownloadBatchResult,
    SeventeenLandsDownloader,
)
from src.data_retrieval.seventeenlands.refs import DataType, SeventeenLandsFileRef

_GAME_MSH_PREMIER = SeventeenLandsFileRef.from_known(
    DataType.GAME, "MSH", "PremierDraft"
)
_DRAFT_MSH_PREMIER = SeventeenLandsFileRef.from_known(
    DataType.DRAFT, "MSH", "PremierDraft"
)
_GAME_WOE_TRAD = SeventeenLandsFileRef.from_known(DataType.GAME, "WOE", "TradDraft")


def _fast_rate_limiter() -> RateLimiter:
    # A real RateLimiter with an effectively unlimited rate, so wait()
    # never actually sleeps in tests.
    return RateLimiter(requests_per_minute=1_000_000_000)


def _make_downloader(raw_data_dir: Path) -> SeventeenLandsDownloader:
    return SeventeenLandsDownloader(
        raw_data_dir=raw_data_dir, rate_limiter=_fast_rate_limiter()
    )


def _mock_streaming_response(chunks: list[bytes]) -> MagicMock:
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.iter_content.return_value = iter(chunks)
    response.__enter__.return_value = response
    response.__exit__.return_value = False
    return response


class TestDestinationPath:
    def test_mirrors_s3_key_structure(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)

        path = downloader._destination_path(_GAME_MSH_PREMIER)

        assert path == tmp_path / "game_data" / "MSH.PremierDraft.csv"


class TestFilterRefs:
    _ALL_REFS = [_GAME_MSH_PREMIER, _DRAFT_MSH_PREMIER, _GAME_WOE_TRAD]

    def test_no_filters_returns_all_refs_in_order(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)

        filtered = downloader._filter_refs(
            self._ALL_REFS, data_types=None, expansions=None, formats=None
        )

        assert filtered == self._ALL_REFS

    def test_filters_by_data_type(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)

        filtered = downloader._filter_refs(
            self._ALL_REFS,
            data_types=[DataType.GAME],
            expansions=None,
            formats=None,
        )

        assert filtered == [_GAME_MSH_PREMIER, _GAME_WOE_TRAD]

    def test_filters_by_expansion(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)

        filtered = downloader._filter_refs(
            self._ALL_REFS, data_types=None, expansions=["MSH"], formats=None
        )

        assert filtered == [_GAME_MSH_PREMIER, _DRAFT_MSH_PREMIER]

    def test_filters_by_format(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)

        filtered = downloader._filter_refs(
            self._ALL_REFS,
            data_types=None,
            expansions=None,
            formats=["TradDraft"],
        )

        assert filtered == [_GAME_WOE_TRAD]

    def test_combines_all_three_filters(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)

        filtered = downloader._filter_refs(
            self._ALL_REFS,
            data_types=[DataType.GAME],
            expansions=["MSH"],
            formats=["PremierDraft"],
        )

        assert filtered == [_GAME_MSH_PREMIER]

    def test_combined_filters_matching_nothing_returns_empty(
        self, tmp_path: Path
    ) -> None:
        downloader = _make_downloader(tmp_path)

        filtered = downloader._filter_refs(
            self._ALL_REFS,
            data_types=[DataType.GAME],
            expansions=["MSH"],
            formats=["TradDraft"],
        )

        assert filtered == []


class TestDownloadOne:
    def test_downloads_and_extracts_to_destination_path(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        content = b"expansion,event_type\nMSH,PremierDraft\n"
        response = _mock_streaming_response([gzip.compress(content)])

        with patch("requests.get", return_value=response) as mock_get:
            result_path = downloader.download_one(_GAME_MSH_PREMIER)

        mock_get.assert_called_once()
        assert mock_get.call_args.args[0] == _GAME_MSH_PREMIER.url
        assert result_path == tmp_path / "game_data" / "MSH.PremierDraft.csv"
        assert result_path.read_bytes() == content

    def test_removes_intermediate_csv_gz_on_success(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        response = _mock_streaming_response([gzip.compress(b"data")])

        with patch("requests.get", return_value=response):
            result_path = downloader.download_one(_GAME_MSH_PREMIER)

        compressed_path = result_path.with_suffix(".csv.gz")
        assert not compressed_path.exists()

    def test_creates_raw_data_dir_if_missing(self, tmp_path: Path) -> None:
        nested_dir = tmp_path / "does" / "not" / "exist"
        downloader = _make_downloader(nested_dir)
        response = _mock_streaming_response([gzip.compress(b"data")])

        with patch("requests.get", return_value=response):
            downloader.download_one(_GAME_MSH_PREMIER)

        assert (nested_dir / "game_data").is_dir()

    def test_raises_on_http_error(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        response = _mock_streaming_response([])
        response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")

        with patch("requests.get", return_value=response):
            with pytest.raises(requests.HTTPError):
                downloader.download_one(_GAME_MSH_PREMIER)

    def test_raises_and_cleans_up_on_invalid_gzip(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        response = _mock_streaming_response([b"not actually gzip data"])

        with patch("requests.get", return_value=response):
            with pytest.raises(gzip.BadGzipFile):
                downloader.download_one(_GAME_MSH_PREMIER)

        destination_path = tmp_path / "game_data" / "MSH.PremierDraft.csv"
        assert not destination_path.exists()
        assert not destination_path.with_suffix(".csv.gz").exists()


class TestDownload:
    def test_downloads_only_filtered_refs(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        response = _mock_streaming_response([gzip.compress(b"data")])

        with patch("requests.get", return_value=response) as mock_get:
            result = downloader.download(
                [_GAME_MSH_PREMIER, _DRAFT_MSH_PREMIER, _GAME_WOE_TRAD],
                data_types=[DataType.GAME],
            )

        assert mock_get.call_count == 2
        assert [outcome.ref for outcome in result.outcomes] == [
            _GAME_MSH_PREMIER,
            _GAME_WOE_TRAD,
        ]
        assert all(outcome.error is None for outcome in result.outcomes)

    def test_empty_refs_downloads_nothing(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)

        with patch("requests.get") as mock_get:
            result = downloader.download([])

        mock_get.assert_not_called()
        assert result == DownloadBatchResult(outcomes=[])

    def test_failed_ref_is_reported_and_loop_continues(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        failing_response = _mock_streaming_response([])
        failing_response.raise_for_status.side_effect = requests.HTTPError(
            "404 Not Found"
        )
        succeeding_response = _mock_streaming_response([gzip.compress(b"data")])

        # Keyed by URL rather than call order: download_to_file() retries
        # internally now, so the failing ref's URL is requested more than
        # once before it's given up on, and a positional side_effect list
        # would hand the second ref's response to one of those retries
        # instead.
        def _get_side_effect(url: str, **kwargs: object) -> MagicMock:
            return failing_response if url == _GAME_MSH_PREMIER.url else succeeding_response

        with patch("requests.get", side_effect=_get_side_effect) as mock_get:
            with patch(
                "src.data_retrieval.seventeenlands.downloader.tqdm.write"
            ) as mock_write:
                result = downloader.download([_GAME_MSH_PREMIER, _GAME_WOE_TRAD])

        assert mock_get.call_count > 1
        assert len(result.outcomes) == 2

        first, second = result.outcomes
        assert first.ref == _GAME_MSH_PREMIER
        assert first.path is None
        assert isinstance(first.error, requests.HTTPError)

        assert second.ref == _GAME_WOE_TRAD
        assert second.path == tmp_path / "game_data" / "WOE.TradDraft.csv"
        assert second.error is None

        # The failing ref didn't stop the second one from being
        # attempted and succeeding, and the failure was reported via
        # tqdm.write (not print), so it doesn't corrupt the progress bar.
        mock_write.assert_called_once()
        written_message = mock_write.call_args.args[0]
        assert _GAME_MSH_PREMIER.url in written_message

    def test_all_refs_failing_returns_all_failures(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        response = _mock_streaming_response([])
        response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")

        with patch("requests.get", return_value=response):
            result = downloader.download([_GAME_MSH_PREMIER, _GAME_WOE_TRAD])

        assert all(outcome.path is None for outcome in result.outcomes)
        assert all(
            isinstance(outcome.error, requests.HTTPError) for outcome in result.outcomes
        )
