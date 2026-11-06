import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Callable
from unittest.mock import patch

import pytest

from src.data_retrieval.rate_limiter import RateLimiter
from src.data_retrieval.tcgjson.downloader import (
    TcgjsonCatalogDownloader,
    parse_release_manifest,
)

_MODULE = "src.data_retrieval.tcgjson.downloader"
_RELEASE = "https://r.test"
_RELEASE_NAME = "20260831T174333Z"
_CATALOG = gzip.compress(b'{"meta": {}, "sets": [], "products": []}')


def _entry(kind: str, file_name: str, body: bytes) -> dict[str, Any]:
    return {
        "type": kind,
        "download_uri": file_name,
        "size": len(body),
        "sha256": hashlib.sha256(body).hexdigest(),
    }


def _manifest(
    *entries: dict[str, Any], generated_at: str = "2026-08-31T17:43:33Z"
) -> str:
    return json.dumps({"generated_at": generated_at, "data": list(entries)})


_LORCANA_ENTRY = _entry("lorcana_catalog_full", "lorcana.full.json.gz", _CATALOG)


def _make_downloader(raw_dir: Path, **kwargs: Any) -> TcgjsonCatalogDownloader:
    # Effectively unlimited, so wait() never sleeps in tests
    kwargs.setdefault("slugs", ("lorcana",))
    return TcgjsonCatalogDownloader(
        RateLimiter(requests_per_minute=1_000_000_000),
        raw_dir,
        release_url=_RELEASE,
        **kwargs,
    )


def _fake_file(bodies: dict[str, bytes], calls: list[str]) -> Callable[..., None]:
    def _download(url: str, destination_path: Path, **_: object) -> None:
        calls.append(url)
        destination_path.write_bytes(bodies[url.rsplit("/", 1)[-1]])

    return _download


def _run(
    raw_dir: Path, manifest: str, bodies: dict[str, bytes], **kwargs: Any
) -> tuple[Path, list[str]]:
    calls: list[str] = []
    with (
        patch(f"{_MODULE}.download_to_string", return_value=manifest),
        patch(f"{_MODULE}.download_to_file", side_effect=_fake_file(bodies, calls)),
    ):
        result = _make_downloader(raw_dir, **kwargs).phase_1()
    return result, calls


class TestPhase1:
    def test_lands_the_catalog_then_the_manifest(self, tmp_path: Path) -> None:
        manifest = _manifest(_LORCANA_ENTRY)
        result, calls = _run(tmp_path, manifest, {"lorcana.full.json.gz": _CATALOG})

        assert result == tmp_path / _RELEASE_NAME
        assert calls == [f"{_RELEASE}/lorcana.full.json.gz"]
        assert (result / "lorcana.full.json.gz").read_bytes() == _CATALOG
        assert (result / "bulk-data.json").read_text() == manifest

    def test_a_landed_catalog_with_the_right_sha256_is_not_fetched(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / _RELEASE_NAME).mkdir()
        (tmp_path / _RELEASE_NAME / "lorcana.full.json.gz").write_bytes(_CATALOG)
        _, calls = _run(tmp_path, _manifest(_LORCANA_ENTRY), {})
        assert calls == []

    def test_a_wrong_sha256_deletes_the_file_and_writes_no_manifest(
        self, tmp_path: Path
    ) -> None:
        entry = {**_LORCANA_ENTRY, "sha256": "0" * 64}
        with pytest.raises(ValueError, match="SHA-256"):
            _run(tmp_path, _manifest(entry), {"lorcana.full.json.gz": _CATALOG})
        assert list((tmp_path / _RELEASE_NAME).iterdir()) == []

    def test_a_wrong_size_deletes_the_file(self, tmp_path: Path) -> None:
        entry = {**_LORCANA_ENTRY, "size": 1}
        with pytest.raises(ValueError, match="bytes"):
            _run(tmp_path, _manifest(entry), {"lorcana.full.json.gz": _CATALOG})
        assert list((tmp_path / _RELEASE_NAME).iterdir()) == []

    def test_a_missing_slug_raises_before_any_download(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="yugioh"):
            _run(
                tmp_path,
                _manifest(_LORCANA_ENTRY),
                {"lorcana.full.json.gz": _CATALOG},
                slugs=("lorcana", "yugioh"),
            )
        assert not (tmp_path / _RELEASE_NAME).exists()


class TestParseReleaseManifest:
    def test_reads_the_release_name_and_files(self) -> None:
        manifest = parse_release_manifest(_manifest(_LORCANA_ENTRY))
        assert manifest.release_name == _RELEASE_NAME
        assert (
            manifest.files["lorcana_catalog_full"].file_name == "lorcana.full.json.gz"
        )
        assert manifest.files["lorcana_catalog_full"].size == len(_CATALOG)

    @pytest.mark.parametrize(
        "text",
        [
            "not json",
            "[]",
            _manifest(_LORCANA_ENTRY, generated_at="2026-08-31"),
            _manifest(_LORCANA_ENTRY, generated_at="../../x"),
            json.dumps({"generated_at": "2026-08-31T17:43:33Z"}),
            _manifest({**_LORCANA_ENTRY, "download_uri": "../evil.gz"}),
            _manifest({**_LORCANA_ENTRY, "download_uri": "a/b.gz"}),
            _manifest({**_LORCANA_ENTRY, "download_uri": "a\\b.gz"}),
            _manifest({**_LORCANA_ENTRY, "size": "12"}),
            _manifest({**_LORCANA_ENTRY, "size": True}),
            _manifest({**_LORCANA_ENTRY, "sha256": "XYZ"}),
            _manifest({"download_uri": "a.gz", "size": 1, "sha256": "0" * 64}),
            _manifest("not an object"),  # type: ignore[arg-type]
        ],
    )
    def test_a_malformed_manifest_is_a_value_error(self, text: str) -> None:
        with pytest.raises(ValueError):
            parse_release_manifest(text)


class TestLatestReleaseDir:
    def test_newest_complete_release(self, tmp_path: Path) -> None:
        for name in ("20260824T105342Z", "20260831T174333Z", "20260907T000000Z"):
            (tmp_path / name).mkdir()
        for name in ("20260824T105342Z", "20260831T174333Z"):
            (tmp_path / name / "bulk-data.json").write_text("{}")
        (tmp_path / "notes").mkdir()
        (tmp_path / "notes" / "bulk-data.json").write_text("{}")

        assert TcgjsonCatalogDownloader.latest_release_dir(tmp_path) == (
            tmp_path / "20260831T174333Z"
        )

    def test_none_complete_raises(self, tmp_path: Path) -> None:
        (tmp_path / "20260907T000000Z").mkdir()
        with pytest.raises(FileNotFoundError):
            TcgjsonCatalogDownloader.latest_release_dir(tmp_path)

    def test_a_missing_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            TcgjsonCatalogDownloader.latest_release_dir(tmp_path / "absent")


def test_empty_slugs_raise(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        _make_downloader(tmp_path, slugs=())
