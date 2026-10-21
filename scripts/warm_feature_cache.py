"""Fill a run config's feature cache ahead of time, without training.

Builds the config's model exactly as scripts/run_training.py would (same
fingerprint, same compute precision), encodes every card of the games in
its top-level `warm_feature_cache:` into every tier durably, logs the
cache's totals and closes it. Useful with a `disk:` tier: the rows
persist, so later runs with the same fingerprint start warm.

Usage (from the project root, in the ROCm venv):

    PYTHONPATH=. python scripts/warm_feature_cache.py configs/training/gwent_contrastive.yaml
    PYTHONPATH=. python scripts/warm_feature_cache.py <config> --set warm_feature_cache=[mtg]
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

from scripts.run_training import (
    build_model,
    closing_feature_cache,
    require_device,
    warm_feature_cache,
)
from src.run_logging import (
    LoggingSettings,
    add_logging_arguments,
    configure_logging,
    logging_settings_from,
    script_log_path,
    script_logger,
)
from src.training.dojo_catalog import DOJO_CATALOG
from src.training.dojo_recipes.card_shelf import CardShelf
from src.training.run_config import (
    RunConfig,
    apply_overrides,
    parse_run_config,
    read_config_document,
)

_logger = script_logger(__file__)


@dataclass(frozen=True)
class CommandLine:
    """The parsed command line: the run config, its --set overrides and
    the logging settings."""

    config: Path
    overrides: tuple[str, ...]
    logging: LoggingSettings


def parse_command_line() -> CommandLine:
    """Parse sys.argv into a CommandLine.

    Inputs: none (reads sys.argv). Output: CommandLine.
    Side effects: argparse prints usage and exits on bad arguments.
    Exceptions: SystemExit from argparse.

    Example:
        >>> parse_command_line().config
        PosixPath('configs/training/gwent_contrastive.yaml')
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("config", type=Path, help="YAML run config")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY.PATH=VALUE",
        help="override one config value (repeatable; later wins)",
    )
    add_logging_arguments(parser)
    args = parser.parse_args()
    return CommandLine(
        config=args.config,
        overrides=tuple(args.overrides),
        logging=logging_settings_from(args, script_log_path(__file__)),
    )


def main() -> int:
    """Warm the config's feature cache; return a process exit code.

    Output: 0 when the cache was warmed, 1 when the config has no feature
        cache or no games to warm.
    Side effects: loads card binders; builds the model on the config's
        device; writes every warmed row to every tier (the disk tier's
        file is created if absent); logs.
    Exceptions: ValueError for a bad config; RuntimeError if the device is
        unavailable; FileNotFoundError for a missing binder.

    Example:
        >>> raise SystemExit(main())
    """
    command_line = parse_command_line()
    configure_logging(command_line.logging)
    document = apply_overrides(
        read_config_document(command_line.config), command_line.overrides
    )
    config = parse_run_config(document, tuple(DOJO_CATALOG))
    if not _has_something_to_warm(config):
        return 1
    require_device(config.device)

    # Build exactly the run's model, so its fingerprint matches the run's
    model = build_model(
        config.model, seed=config.plan.seed, compute_precision=config.limits.precision
    ).to(config.device)
    with closing_feature_cache(model):
        warm_feature_cache(model, config, CardShelf())
    return 0


def _has_something_to_warm(config: RunConfig) -> bool:
    """Whether config names a feature cache and games to warm; logs why not.

    Private helper - single consumer is main.
    Inputs: config. Output: bool. Side effects: one ERROR log line when
        False; a WARNING when there is no disk tier (nothing persists).
        Exceptions: none.
    """
    cache = config.model.feature_cache
    if cache is None:
        _logger.error("the config has no model.feature_cache section; nothing to warm")
        return False
    if not config.warm_feature_cache:
        _logger.error("the config's warm_feature_cache lists no games; nothing to warm")
        return False
    if cache.disk is None:
        _logger.warning("no disk tier: the warmed rows are lost when this script exits")
    return True


if __name__ == "__main__":
    raise SystemExit(main())
