"""Runs src/data_retrieval downloaders — one at a time, or all of them
at once, each in its own terminal window.

Every downloader here follows Downloader's phase_1()/phase_2()
convention (phase_2() is a no-op for a source with nothing further to
fetch, so calling both unconditionally is always safe) — except
`seventeenlands`, which predates that base class and exposes a
different shape (download(), no phase_1/phase_2) that this script's
own README-documented exception carries forward rather than papering
over. See src/data_retrieval/README.md for what each source actually
does.

Several of these (spire_codex_runs, play_gwent, pitchstack,
fabtcg_decklists, sts_gg, seventeenlands) are 10+ hour crawls, and
almost every downloader below reports its own progress via tqdm.
Running several in one process/terminal makes their progress bars fight
over the same lines, so `--all` instead spawns one subprocess per
downloader, each in its own new console window — tqdm then has a real,
uncontested terminal to draw in, and any one source can be watched,
closed, or killed without touching the others. Every downloader is
independently resumable (skips whatever it already wrote to disk), so
closing a window and re-running just that `--source` later is safe.

Usage (from the project root):

    PYTHONPATH=. python3 scripts/run_data_retrieval.py --list
    PYTHONPATH=. python3 scripts/run_data_retrieval.py --source seventeenlands
    PYTHONPATH=. python3 scripts/run_data_retrieval.py
    PYTHONPATH=. python3 scripts/run_data_retrieval.py --all

With no arguments (equivalently, `--all`), every downloader is spawned
in its own window and this script returns immediately — it does not
wait for them to finish. `--source <name>` runs just that one
downloader in the current terminal instead (the old single-downloader
dev/test workflow this script used to hardcode via commented-out
lines).

Terminal-per-downloader is implemented for Windows (a new console via
CREATE_NEW_CONSOLE) and best-effort for macOS (Terminal.app via
osascript) and Linux (whichever of gnome-terminal/konsole/xterm/
x-terminal-emulator is on PATH) — only the Windows path has actually
been exercised. If no terminal emulator can be found, that downloader
falls back to running inline in the current terminal instead of being
skipped.
"""

import argparse
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Callable

from src.data_retrieval.cardvault_fabtcg.card_downloader import (
    CardVaultFabtcgCardDownloader,
)
from src.data_retrieval.dominiontabs.downloader import DominionTabsCardDownloader
from src.data_retrieval.download_utils import download_to_string
from src.data_retrieval.downloader import Downloader
from src.data_retrieval.fabtcg_decklists.downloader import FabtcgDecklistDownloader
from src.data_retrieval.gwent_one.downloader import GwentOneDownloader
from src.data_retrieval.hearthstonejson.downloader import HearthstoneJsonDownloader
from src.data_retrieval.pitchstack.downloader import PitchstackDeckDownloader
from src.data_retrieval.play_gwent.downloader import PlayGwentDownloader
from src.data_retrieval.pokemon_tcg.downloader import PokemonTcgDataDownloader
from src.data_retrieval.rate_limiter import RateLimiter
from src.data_retrieval.scryfall.downloader import ScryfallOracleDownloader
from src.data_retrieval.seventeenlands.downloader import SeventeenLandsDownloader
from src.data_retrieval.spire_codex.card_downloader import SpireCodexCardDownloader
from src.data_retrieval.spire_codex.run_downloader import SpireCodexRunDownloader
from src.data_retrieval.sts2runs.downloader import STS2RunsDownloader
from src.data_retrieval.sts_gg.run_downloader import STSGGRunDownloader

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Stable, non-dated source URLs these two downloaders require but don't
# hardcode themselves (their classes are generic over "which repo/host",
# so the concrete URL belongs at the call site, not in the class) — see
# src/data_retrieval/README.md's "How to run" section, which uses the
# same two values.
_POKEMON_TCG_ZIP_URL = (
    "https://api.github.com/repos/PokemonTCG/pokemon-tcg-data/zipball"
)
_HEARTHSTONEJSON_LISTING_URL = "https://api.hearthstonejson.com/v1/"
_SCRYFALL_BULK_DATA_API = "https://api.scryfall.com/bulk-data"


def _current_scryfall_oracle_cards_url() -> str:
    """Look up today's oracle-cards dump URL from Scryfall's bulk-data
    API, rather than hardcoding one — unlike every other source here,
    Scryfall's dump URL is timestamped and changes with every release,
    so a hardcoded default would silently go stale (see
    ScryfallOracleDownloader's docstring).

    Scryfall 400s the default python-requests User-Agent on this
    endpoint (its "generic_user_agent" rule) — same class of issue as
    fabtcg_decklists' WAF, worked around the same way (see
    download_to_string's headers param).
    """
    payload = json.loads(
        download_to_string(
            _SCRYFALL_BULK_DATA_API,
            headers={"User-Agent": "nocab-card-embeddings/1.0"},
        )
    )
    for entry in payload["data"]:
        if entry["type"] == "oracle_cards":
            return str(entry["jsonl_download_uri"])
    raise RuntimeError(f"{_SCRYFALL_BULK_DATA_API} returned no 'oracle_cards' entry")


def _run_standard(downloader: Downloader) -> None:
    """Run any ordinary Downloader subclass to completion and report
    where its output landed."""
    downloader.phase_1()
    result_path = downloader.phase_2()
    print(f"\nDone: {result_path}")


def run_scryfall() -> None:
    url = _current_scryfall_oracle_cards_url()
    print(f"Scryfall oracle-cards dump: {url}")
    _run_standard(ScryfallOracleDownloader(url))


def run_pokemon_tcg() -> None:
    _run_standard(PokemonTcgDataDownloader(_POKEMON_TCG_ZIP_URL))


def run_hearthstonejson() -> None:
    _run_standard(HearthstoneJsonDownloader(_HEARTHSTONEJSON_LISTING_URL))


def run_spire_codex_cards() -> None:
    _run_standard(SpireCodexCardDownloader())


def run_spire_codex_runs() -> None:
    _run_standard(SpireCodexRunDownloader())


def run_gwent_one() -> None:
    _run_standard(GwentOneDownloader())


def run_sts_gg() -> None:
    _run_standard(STSGGRunDownloader())


def run_sts2runs() -> None:
    _run_standard(STS2RunsDownloader())


def run_play_gwent() -> None:
    _run_standard(PlayGwentDownloader())


def run_cardvault_fabtcg() -> None:
    _run_standard(CardVaultFabtcgCardDownloader())


def run_dominiontabs() -> None:
    _run_standard(DominionTabsCardDownloader())


def run_pitchstack() -> None:
    _run_standard(PitchstackDeckDownloader())


def run_fabtcg_decklists() -> None:
    _run_standard(FabtcgDecklistDownloader())


def run_seventeenlands() -> None:
    # Doesn't subclass Downloader and has no phase_1()/phase_2() — see
    # this module's docstring and src/data_retrieval/README.md.
    downloader = SeventeenLandsDownloader(
        rate_limiter=RateLimiter(requests_per_minute=60)
    )
    result = downloader.download()
    failed = [outcome for outcome in result.outcomes if outcome.error is not None]
    print(
        f"\nDone: {len(result.outcomes) - len(failed)} succeeded, {len(failed)} failed"
    )


DOWNLOADERS: dict[str, Callable[[], None]] = {
    "cardvault_fabtcg": run_cardvault_fabtcg,
    "dominiontabs": run_dominiontabs,
    "fabtcg_decklists": run_fabtcg_decklists,
    "gwent_one": run_gwent_one,
    "hearthstonejson": run_hearthstonejson,
    "pitchstack": run_pitchstack,
    "play_gwent": run_play_gwent,
    "pokemon_tcg": run_pokemon_tcg,
    "scryfall": run_scryfall,
    "seventeenlands": run_seventeenlands,
    "spire_codex_cards": run_spire_codex_cards,
    "spire_codex_runs": run_spire_codex_runs,
    "sts2runs": run_sts2runs,
    "sts_gg": run_sts_gg,
}


def _spawn_worker(name: str) -> subprocess.Popen:
    """Launch `--source name --pause-on-exit` in its own new terminal
    window, platform-dependently (see module docstring)."""
    args = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--source",
        name,
        "--pause-on-exit",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)

    system = platform.system()

    if system == "Windows":
        return subprocess.Popen(
            args,
            cwd=PROJECT_ROOT,
            env=env,
            creationflags=subprocess.CREATE_NEW_CONSOLE,
        )

    if system == "Darwin":
        shell_command = " ".join(shlex.quote(part) for part in args)
        apple_script = (
            f'tell application "Terminal" to do script '
            f'"cd {shlex.quote(str(PROJECT_ROOT))} && {shell_command}"'
        )
        return subprocess.Popen(["osascript", "-e", apple_script], env=env)

    for terminal, exec_args in (
        ("gnome-terminal", ["--"]),
        ("konsole", ["-e"]),
        ("xterm", ["-e"]),
        ("x-terminal-emulator", ["-e"]),
    ):
        if shutil.which(terminal):
            return subprocess.Popen(
                [terminal, *exec_args, *args], cwd=PROJECT_ROOT, env=env
            )

    print(
        f"No supported terminal emulator found — running {name!r} inline instead.",
        file=sys.stderr,
    )
    return subprocess.Popen(args, cwd=PROJECT_ROOT, env=env)


def run_all_in_separate_terminals() -> None:
    names = sorted(DOWNLOADERS)
    print(f"Launching {len(names)} downloaders, each in its own terminal window:")
    for name in names:
        process = _spawn_worker(name)
        print(f"  {name}: pid {process.pid}")
    print(
        "\nThis script does not wait for them — each window runs "
        "independently and can be closed or re-run on its own; every "
        "downloader resumes from whatever it already wrote to disk."
    )


def run_one(name: str, *, pause_on_exit: bool) -> None:
    print(f"=== {name} ===")
    exit_code = 0
    try:
        DOWNLOADERS[name]()
    except Exception:
        traceback.print_exc()
        exit_code = 1

    if pause_on_exit:
        input("\nPress Enter to close this window...")
    sys.exit(exit_code)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--list", action="store_true", help="List available downloader names and exit."
    )
    parser.add_argument(
        "--source",
        choices=sorted(DOWNLOADERS),
        help="Run just this one downloader in the current terminal.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run every downloader, each in its own new terminal window (the default "
        "when no other flag is given).",
    )
    parser.add_argument(
        "--pause-on-exit",
        action="store_true",
        help="Wait for a keypress before closing. Set automatically on the workers "
        "--all spawns, so a finished (or crashed) window stays visible; harmless to "
        "pass by hand too.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list:
        for name in sorted(DOWNLOADERS):
            print(name)
        return

    if args.source:
        run_one(args.source, pause_on_exit=args.pause_on_exit)
        return

    run_all_in_separate_terminals()


if __name__ == "__main__":
    main()
