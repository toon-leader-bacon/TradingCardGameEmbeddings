"""Mirrors one Hugging Face dataset repo's top-level files onto disk.

See src/data_retrieval/README.md for this container's scope: fetch and
land raw files on disk, no parsing or filtering. Reading the files is
data_refinement's job.

THE INTERFACE: the Hub's public JSON API, no key needed.

    1. GET {hub}/api/datasets/{repo_id} -> the repo's current commit
       "sha". Every later request names that sha, so a repo updated
       mid-run can't mix two revisions in one mirror.
    2. GET {hub}/api/datasets/{repo_id}/tree/{sha} -> the top-level
       listing, non-recursive: entries {"type": "file" | "directory",
       "path", "size", "oid", "lfs": {"oid", "size"} (LFS files only)}.
       Long listings are paged with a Link: <...>; rel="next" header.
    3. GET {hub}/datasets/{repo_id}/resolve/{sha}/{path} -> a file's
       bytes (a redirect to a CDN, which download_to_file follows).

phase_1() mirrors in place into raw_data_dir (default
raw_data_dir_for(repo_id) = data/raw/huggingface/<owner>__<name>/), not
one directory per revision: a daily-updated repo changes only a few of
its files, so a re-run downloads those alone. It deletes revision.json
first, skips each file whose local copy already matches the listing's
digest (FileDigest), downloads and checks the rest, deletes local files
the listing no longer names, and writes revision.json last. A
directory holding revision.json is therefore a complete mirror of the
sha it names (is_complete_mirror); a crashed run leaves none, so the
mirror reads as absent until a run finishes. The directory belongs to
the downloader: a caller-supplied raw_data_dir is pruned the same way.

Paths come from downloaded data, so only flat file names matching
_FILE_NAME are mirrored; directory entries, other names and
.gitattributes are skipped (each with a warning, except
.gitattributes). There is no phase 2.
"""

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar

from tqdm import tqdm

from src.data_retrieval.download_utils import (
    RESEARCH_USER_AGENT,
    download_to_file,
    paced_get,
    require_int,
    require_str,
    sha256_of,
)
from src.data_retrieval.downloader import Downloader
from src.data_retrieval.rate_limiter import RateLimiter

_logger = logging.getLogger(__name__)

_DEFAULT_HUB_URL = "https://huggingface.co"
_HEADERS = {"User-Agent": RESEARCH_USER_AGENT}
_DOWNLOAD_TIMEOUT_SECONDS = 600
# One part of an "owner/name" repo id, and a mirrored file's name: no
# separators, so neither can leave the mirror directory
_NAME_PART = r"[A-Za-z0-9][A-Za-z0-9._-]*"
_REPO_ID = re.compile(rf"({_NAME_PART})/({_NAME_PART})")
# A file name may also start with "_" ("_last_deck_num.txt"), never "."
_FILE_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9._-]*")
_SKIPPED_SILENTLY = frozenset({".gitattributes"})
_FILE_ENTRY_TYPE = "file"
# The Link header's next page: <https://...>; rel="next"
_NEXT_PAGE_LINK = re.compile(r'<([^>]+)>;\s*rel="next"')
_COMMIT_SHA = re.compile(r"[0-9a-f]{40}")
_HASH_CHUNK_BYTES = 1 << 20


class DigestKind(Enum):
    """How the Hub names a file's content: an LFS file by the SHA-256
    of its bytes, any other file by its git blob SHA-1,
    sha1(b"blob <len>\\0" + bytes)."""

    SHA256 = "sha256"
    GIT_BLOB_SHA1 = "git_blob_sha1"

    def hex_of(self, path: Path) -> str:
        """path's digest of this kind, lowercase hex.

        Inputs: path. Output: str. Side effects: reads path.
        Exceptions: OSError if path can't be read.

        Example:
            >>> DigestKind.GIT_BLOB_SHA1.hex_of(Path("00000001.csv"))
            'c9f7d74a...'
        """
        if self is DigestKind.SHA256:
            return sha256_of(path)
        return _git_blob_sha1_of(path)


@dataclass(frozen=True)
class FileDigest:
    """The content digest the listing gives for one file."""

    kind: DigestKind
    hex: str


@dataclass(frozen=True)
class HubFile:
    """One top-level file of a repo revision, parsed from the tree
    listing: its name (a _FILE_NAME), size in bytes and digest."""

    name: str
    size: int
    digest: FileDigest

    def local_mismatch(self, path: Path) -> str | None:
        """Why path isn't this file (missing, wrong size, wrong
        digest), or None when it is. Checks the cheap things first, so
        a file whose size changed is never hashed. Both the skip check
        and the post-download check use it.

        Inputs: path. Output: str (the reason) | None.
        Side effects: reads path (only when the size matches).
        Exceptions: OSError if path exists but can't be read.

        Example:
            >>> hub_file.local_mismatch(Path("00000001.csv"))
            'size 390000, expected 390885'
        """
        if not path.is_file():
            return "missing"
        size = path.stat().st_size
        if size != self.size:
            return f"size {size}, expected {self.size}"
        actual = self.digest.kind.hex_of(path)
        if actual != self.digest.hex:
            return f"{self.digest.kind.value} {actual}, expected {self.digest.hex}"
        return None


@dataclass(frozen=True)
class HubRevision:
    """A pinned repo revision: its commit sha and the files mirrored
    from it, in listing order."""

    sha: str
    files: tuple[HubFile, ...]


class HuggingFaceDatasetDownloader(Downloader):
    """Mirrors one Hugging Face dataset repo's top-level files.

    Repo-agnostic. Used by scripts/run_data_retrieval.py (to mirror) and
    by the stages that read a mirror (raw_data_dir_for,
    require_complete_mirror): card_binder/ygoprodeck/ and
    deck_box/ygoprodeck_decks/.
    """

    # The root of every repo's directory (see raw_data_dir_for)
    DEFAULT_RAW_DATA_DIR: ClassVar[Path] = Path("data/raw/huggingface")
    REVISION_FILENAME: ClassVar[str] = "revision.json"

    def __init__(
        self,
        repo_id: str,
        rate_limiter: RateLimiter | None = None,
        raw_data_dir: Path | None = None,
        hub_url: str = _DEFAULT_HUB_URL,
    ) -> None:
        """
        Inputs:
            repo_id: the dataset repo, "owner/name".
            rate_limiter: see Downloader.__init__ (60/min by default).
            raw_data_dir: see Downloader.__init__. Defaults to
                raw_data_dir_for(repo_id).
            hub_url: the Hub's base URL, no trailing slash.
        Output: none (constructor).
        Side effects: none; no I/O until phase_1().
        Exceptions: ValueError if repo_id isn't "owner/name" with each
            part a plain name.

        Example:
            >>> HuggingFaceDatasetDownloader("feregrino/yugioh-decks").raw_data_dir.name
            'feregrino__yugioh-decks'
        """
        directory = (
            raw_data_dir if raw_data_dir is not None else self.raw_data_dir_for(repo_id)
        )
        super().__init__(rate_limiter, directory)
        self.repo_id = repo_id
        self.hub_url = hub_url

    @classmethod
    def raw_data_dir_for(cls, repo_id: str) -> Path:
        """DEFAULT_RAW_DATA_DIR / repo_id with "/" as "__", the
        directory a repo is mirrored into by default.

        Inputs: repo_id ("owner/name"). Output: Path.
        Side effects: none.
        Exceptions: ValueError if repo_id isn't "owner/name".

        Example:
            >>> HuggingFaceDatasetDownloader.raw_data_dir_for("feregrino/yugioh-cards")
            Path('data/raw/huggingface/feregrino__yugioh-cards')
        """
        match = _REPO_ID.fullmatch(repo_id)
        if match is None:
            raise ValueError(f"not an owner/name Hugging Face repo id: {repo_id!r}")
        return cls.DEFAULT_RAW_DATA_DIR / f"{match.group(1)}__{match.group(2)}"

    @classmethod
    def is_complete_mirror(cls, raw_data_dir: Path) -> bool:
        """Whether raw_data_dir holds revision.json, i.e. a finished
        phase_1() mirrored into it.

        Inputs: raw_data_dir. Output: bool.
        Side effects: checks the file system. Exceptions: none.

        Example:
            >>> HuggingFaceDatasetDownloader.is_complete_mirror(
            ...     Path("data/raw/huggingface/feregrino__yugioh-decks"))
            True
        """
        return (raw_data_dir / cls.REVISION_FILENAME).is_file()

    @classmethod
    def require_complete_mirror(cls, raw_data_dir: Path) -> None:
        """Raise unless is_complete_mirror(raw_data_dir): the check every
        reader of a mirror makes before reading it.

        Inputs: raw_data_dir. Output: none.
        Side effects: checks the file system.
        Exceptions: FileNotFoundError naming the missing revision.json.

        Example:
            >>> HuggingFaceDatasetDownloader.require_complete_mirror(
            ...     Path("data/raw/huggingface/feregrino__yugioh-cards"))
        """
        if not cls.is_complete_mirror(raw_data_dir):
            raise FileNotFoundError(
                f"no complete Hugging Face mirror at {raw_data_dir} (no "
                f"{cls.REVISION_FILENAME}; run scripts/run_data_retrieval.py)"
            )

    def _run_phase_1(self) -> Path:
        """Mirror the repo's current revision into raw_data_dir.

        Inputs: none (uses self.repo_id, self.hub_url).
        Output: self.raw_data_dir.
        Side effects: paced requests (two or more for the listing, one
            per changed file); creates raw_data_dir; deletes
            revision.json first and writes it last; downloads changed
            files; deletes local files the listing doesn't name; shows
            a tqdm bar; logs skipped listing entries.
        Exceptions: requests.RequestException if a request still fails
            after retries; ValueError if the listing keeps no file or
            the API's JSON isn't the expected shape, or a download's
            size or digest is wrong (the file is deleted first);
            TypeError if a listing entry's fields aren't the expected
            types.

        Example:
            >>> HuggingFaceDatasetDownloader("feregrino/yugioh-cards").phase_1()
        """
        # Pin the revision and list its files before touching the disk
        sha = self._fetch_revision_sha()
        revision = HubRevision(sha, self._fetch_files(sha))
        if not revision.files:
            raise ValueError(f"{self.repo_id}@{revision.sha}: no file to mirror")

        # Mark the mirror incomplete, then bring each file up to date
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        (self.raw_data_dir / self.REVISION_FILENAME).unlink(missing_ok=True)
        for hub_file in tqdm(revision.files, desc=self.repo_id, unit="file"):
            if hub_file.local_mismatch(self.raw_data_dir / hub_file.name) is None:
                continue
            self._download_file(revision.sha, hub_file)

        # Drop what the revision no longer has; mark the mirror complete
        self._delete_unlisted_files(revision)
        self._write_revision(revision)
        return self.raw_data_dir

    def _fetch_revision_sha(self) -> str:
        """The repo's current commit sha.

        Inputs: none. Output: the sha (hex).
        Side effects: one paced request.
        Exceptions: requests.RequestException after retries;
            ValueError if the body has no string "sha".
        """
        url = f"{self.hub_url}/api/datasets/{self.repo_id}"
        response = paced_get(url, rate_limiter=self.rate_limiter, headers=_HEADERS)
        body = _json_body(response.text, url)
        sha = body.get("sha") if isinstance(body, dict) else None
        if not isinstance(sha, str) or _COMMIT_SHA.fullmatch(sha) is None:
            raise ValueError(f"{url}: no commit sha in the response")
        return sha

    def _fetch_files(self, sha: str) -> tuple[HubFile, ...]:
        """Every mirrorable top-level file of revision sha, following
        the listing's next-page links (a loop, not recursion).

        Inputs: sha. Output: the kept files, in listing order.
        Side effects: one paced request per listing page; logs a
            warning per skipped entry (except _SKIPPED_SILENTLY).
        Exceptions: requests.RequestException after retries;
            ValueError if a page isn't a JSON list of entries; TypeError
            if a file entry's fields aren't the expected types.
        """
        result: list[HubFile] = []
        url: str | None = f"{self.hub_url}/api/datasets/{self.repo_id}/tree/{sha}"

        # Walk the pages until one has no next link
        while url is not None:
            response = paced_get(url, rate_limiter=self.rate_limiter, headers=_HEADERS)
            entries = _json_body(response.text, url)
            if not isinstance(entries, list):
                raise ValueError(f"{url}: the listing isn't a JSON list")
            for entry in entries:
                hub_file = _hub_file_from_entry(entry)
                if hub_file is not None:
                    result.append(hub_file)
            url = _next_page_url(response.headers.get("Link", ""))
        return tuple(result)

    def _download_file(self, sha: str, hub_file: HubFile) -> None:
        """Download one file of revision sha into raw_data_dir and check
        it (HubFile.local_mismatch).

        Inputs: sha, hub_file. Output: none.
        Side effects: one paced download; writes the file; deletes it
            on a size or digest mismatch.
        Exceptions: requests.RequestException after retries;
            ValueError on a mismatch.
        """
        destination_path = self.raw_data_dir / hub_file.name
        self.rate_limiter.wait()
        download_to_file(
            f"{self.hub_url}/datasets/{self.repo_id}/resolve/{sha}/{hub_file.name}",
            destination_path,
            headers=_HEADERS,
            timeout_seconds=_DOWNLOAD_TIMEOUT_SECONDS,
        )
        mismatch = hub_file.local_mismatch(destination_path)
        if mismatch is not None:
            destination_path.unlink(missing_ok=True)
            raise ValueError(f"{self.repo_id} {hub_file.name}: {mismatch}; deleted")

    def _delete_unlisted_files(self, revision: HubRevision) -> None:
        """Delete every file in raw_data_dir that revision doesn't list
        (revision.json is already gone; subdirectories are left alone).

        Inputs: revision. Output: none.
        Side effects: deletes files; logs each at INFO.
        Exceptions: OSError on a failed delete.

        Names compare case-sensitively: on a case-insensitive file
        system a listed name differing from a local one only in case
        would delete the download. No mirrored repo has such names.
        """
        listed = {hub_file.name for hub_file in revision.files}
        for path in sorted(self.raw_data_dir.iterdir()):
            if path.is_file() and path.name not in listed:
                _logger.info(
                    "%s: deleting %s, not in %s", self.repo_id, path.name, revision.sha
                )
                path.unlink()

    def _write_revision(self, revision: HubRevision) -> None:
        """Write revision.json: {"repo_id", "sha", "files": [names]}.

        Inputs: revision. Output: none.
        Side effects: writes raw_data_dir/revision.json.
        Exceptions: OSError on a write failure.
        """
        payload = {
            "repo_id": self.repo_id,
            "sha": revision.sha,
            "files": [hub_file.name for hub_file in revision.files],
        }
        (self.raw_data_dir / self.REVISION_FILENAME).write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )


def _hub_file_from_entry(entry: Any) -> HubFile | None:
    """Parse one tree-listing entry; None for an entry not mirrored (a
    directory, .gitattributes, revision.json's name, a name that isn't
    a plain file name).

    Inputs: entry (decoded JSON). Output: HubFile | None.
    Side effects: logs a warning for a skipped entry other than
        _SKIPPED_SILENTLY.
    Exceptions: TypeError if the entry isn't an object, or a file
        entry's path, size or oid (or its lfs oid) isn't the expected
        type (download_utils' require_str / require_int).
    """
    if not isinstance(entry, dict):
        raise TypeError(f"expected a listing entry object, got {entry!r}")
    name = require_str(entry.get("path"))
    if name in _SKIPPED_SILENTLY:
        return None
    if entry.get("type") != _FILE_ENTRY_TYPE:
        _logger.warning("Hugging Face listing: skipping %s %r", entry.get("type"), name)
        return None
    if (
        _FILE_NAME.fullmatch(name) is None
        or name == HuggingFaceDatasetDownloader.REVISION_FILENAME
    ):
        _logger.warning(
            "Hugging Face listing: skipping file %r (not a plain name)", name
        )
        return None

    # An LFS file is named by its bytes' SHA-256, any other by its blob oid
    lfs = entry.get("lfs")
    if isinstance(lfs, dict):
        digest = FileDigest(DigestKind.SHA256, require_str(lfs.get("oid")))
    else:
        digest = FileDigest(DigestKind.GIT_BLOB_SHA1, require_str(entry.get("oid")))
    return HubFile(name=name, size=require_int(entry.get("size")), digest=digest)


def _next_page_url(link_header: str) -> str | None:
    """The rel="next" URL of a Link header, or None.

    Inputs: link_header (may be ""). Output: str | None.
    Side effects: none. Exceptions: none.
    """
    match = _NEXT_PAGE_LINK.search(link_header)
    return match.group(1) if match else None


def _git_blob_sha1_of(path: Path) -> str:
    """path's git blob SHA-1 (sha1 of b"blob <size>\\0" + bytes),
    lowercase hex, read in chunks.

    Inputs: path. Output: str. Side effects: reads path.
    Exceptions: OSError if path can't be read.
    """
    digest = hashlib.sha1(f"blob {path.stat().st_size}\0".encode("ascii"))
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_body(text: str, url: str) -> Any:
    """Decode an API body, naming url in the error.

    Inputs: text, url. Output: the decoded JSON.
    Side effects: none. Exceptions: ValueError if text isn't JSON.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"{url}: not JSON ({error})") from error
