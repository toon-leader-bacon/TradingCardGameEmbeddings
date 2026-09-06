"""
PYTHONPATH=. python3 scripts/run_data_retrieval.py
"""
from pathlib import Path

from src.data_retrieval.play_gwent.downloader import \
    PlayGwentDownloader as gwent_deck_downloader
from src.data_retrieval.rate_limiter import RateLimiter
from src.data_retrieval.sts_gg.run_downloader import \
    STSGGRunDownloader as sts_gg_run_downloader
from src.data_retrieval.sts2runs.downloader import \
    STS2RunsDownloader as sts2runs_downloader
from src.data_retrieval.pitchstack.downloader import \
    PitchstackDeckDownloader as pitchstack_downloader


def main() -> None:
    # downloader = gwent_deck_downloader(
    #     rate_limiter=RateLimiter(requests_per_minute=60),
    #     output_dir=Path("data/raw/play_gwent"),
    # )
    # downloader = sts_gg_run_downloader(
    #     rate_limiter=RateLimiter(requests_per_minute=60),
    #     output_dir=Path("data/raw/sts_gg"),
    # )
    # downloader = sts2runs_downloader(
    #     raw_data_dir=Path("data/raw/sts2runs"),
    # )
    downloader = pitchstack_downloader(
        rate_limiter=RateLimiter(requests_per_minute=60),
        output_dir=Path("data/raw/pitchstack"),
    )
    downloader.phase_1()
    downloader.phase_2()


if __name__ == "__main__":
    main()
