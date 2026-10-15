"""Fit a token whitening for a frozen text encoder and save it.

    PYTHONPATH=. python scripts/fit_whitening.py \
        --holdout-from configs/training/<run>.yaml \
        --out data/whitening/modernbert_tokens.pt

Only TRAIN-tier cards of the run's holdout are sampled (held-out games
never), so the stats can be used by any run with that same holdout; a run
with another holdout refuses them. Without --holdout-from every card may
be sampled (HoldoutSpec.no_holdout()), for offline analysis only. A run
uses the file through `model.enrichment: {kind: whitening, stats_path:
...}` (src/training/config_parsing/model_parsing.py).
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch

from src.data_refinement.card_binder.card_binder import CardBinder
from src.encoder_model.reference_card_pipelines import DEFAULT_PRETRAINED_CHECKPOINT
from src.encoder_model.text_encoder import PretrainedTextEncoder
from src.encoder_model.whitening.calibration import (
    CalibrationSampleOptions,
    WhiteningFitOptions,
    fit_token_whitening,
    sample_calibration_cards,
)
from src.run_logging import (
    LoggingSettings,
    add_logging_arguments,
    configure_logging,
    logging_settings_from,
    script_log_path,
    script_logger,
)
from src.schema.game_id import GameId
from src.schema.holdout import HoldoutSpec
from src.training.dojo_catalog import DOJO_CATALOG
from src.training.dojo_recipes.card_shelf import CardShelf
from src.training.run_config import RunConfig, parse_run_config, read_config_document

_logger = script_logger(__file__)


@dataclass(frozen=True)
class CommandLine:
    """Parsed arguments: see parse_command_line()."""

    checkpoint: str | None  # None: the run config's, else the default
    holdout_from: Path | None
    games: tuple[GameId, ...] | None  # None: every game with a binder
    cards_per_game: int
    components: int | None
    device: str
    out: Path
    logging: LoggingSettings


def parse_command_line() -> CommandLine:
    """Inputs: sys.argv. Output: CommandLine.
    Side effects: none (argparse exits on --help or bad arguments).
    Exceptions: SystemExit from argparse.

    Example:
        >>> parse_command_line().cards_per_game
        1000
    """
    parser = argparse.ArgumentParser(description="Fit and save a token whitening.")
    parser.add_argument(
        "--checkpoint",
        help="default: the --holdout-from config's model.checkpoint, else "
        + DEFAULT_PRETRAINED_CHECKPOINT,
    )
    parser.add_argument(
        "--holdout-from",
        type=Path,
        help="a run config: sample its TRAIN tier only, fit its text encoder",
    )
    parser.add_argument(
        "--games",
        nargs="+",
        choices=[game.value for game in GameId],
        help="default: every game with a card binder, minus held-out games",
    )
    parser.add_argument("--cards-per-game", type=int, default=1000)
    parser.add_argument(
        "--components", type=int, help="dimensions to keep (default: all)"
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, required=True)
    add_logging_arguments(parser)
    args = parser.parse_args()
    return CommandLine(
        checkpoint=args.checkpoint,
        holdout_from=args.holdout_from,
        games=tuple(GameId(g) for g in args.games) if args.games else None,
        cards_per_game=args.cards_per_game,
        components=args.components,
        device=args.device,
        out=args.out,
        logging=logging_settings_from(args, script_log_path(__file__)),
    )


def main() -> int:
    """Fit and save; return a process exit code.

    Output: 0.
    Side effects: loads card binders, may download the checkpoint, runs
        the text encoder on the device, writes the stats file, logs a
        summary.
    Exceptions: ValueError for a bad config or an empty sample;
        FileNotFoundError for a missing binder.

    Example:
        >>> raise SystemExit(main())
    """
    command_line = parse_command_line()
    configure_logging(command_line.logging)

    # The run's holdout and text encoder, or no holdout and the default
    config = _run_config_of(command_line.holdout_from)
    holdout = config.plan.holdout if config else HoldoutSpec.no_holdout()
    checkpoint = command_line.checkpoint or (
        config.model.checkpoint if config else DEFAULT_PRETRAINED_CHECKPOINT
    )
    games = command_line.games or _games_with_binders(holdout)
    sample_options = CalibrationSampleOptions(
        holdout=holdout, games=games, cards_per_game=command_line.cards_per_game
    )

    # Sample, fit, save
    sample = sample_calibration_cards(
        CardShelf().card_lookup(list(games)), sample_options
    )
    encoder = PretrainedTextEncoder(checkpoint).to(command_line.device)
    stats = fit_token_whitening(
        encoder, sample, WhiteningFitOptions(components=command_line.components)
    )
    stats.save(command_line.out)
    _logger.info(
        "Fitted %d -> %d token whitening on %d cards (%s); saved %s",
        stats.input_dim,
        stats.output_dim,
        stats.card_count,
        ", ".join(g.value for g in games),
        command_line.out,
    )
    return 0


def _run_config_of(config_path: Path | None) -> RunConfig | None:
    """Private helper - single consumer is main.

    Inputs: config_path (a run config) or None.
    Output: the parsed RunConfig, or None.
    Side effects: reads the config. Exceptions: ValueError for a bad config.
    """
    if config_path is None:
        return None
    return parse_run_config(read_config_document(config_path), tuple(DOJO_CATALOG))


def _games_with_binders(holdout: HoldoutSpec) -> tuple[GameId, ...]:
    """Private helper - single consumer is main.

    Inputs: holdout. Output: every GameId whose card binder file exists,
        minus holdout.held_out_games, in GameId order.
    Side effects: checks files exist. Exceptions: none.
    """
    return tuple(
        game
        for game in GameId
        if game not in holdout.held_out_games
        and CardBinder.default_output_path(game).exists()
    )


if __name__ == "__main__":
    raise SystemExit(main())
