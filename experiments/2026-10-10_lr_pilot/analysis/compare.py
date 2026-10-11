"""Compare normalized TEST loss per round across the archived baseline run
and the LR pilots, from their console logs. argv: log paths."""

import re
import statistics
import sys
from pathlib import Path

HELD_OUT = {
    "cross_game.rarity_tier",
    "cardvault_fabtcg.pitch_mask",
    "fabtcg_decklists.hero_masked_from_deck",
    "hearthstonejson.class_mask",
    "isotropic.kingdom_veto_prediction",
    "pokemon_tcg.hp_regression",
    "scryfall.cmc_regression",
    "seventeenlands_game_data.deck_win_prediction",
    "seventeenlands_replay_data.average_turn_cast",
    "spire_codex.card_type_mask",
    "sts2_runs.card_reward_pick",
    "sts2_runs.card_win_rate",
    "sts2_runs.card_win_rate_at_act2",
}
ROUND = re.compile(r"\[train\] round (\d+) \(step \d+\) test loss: (.*?) \|")
ENTRY = re.compile(r"([\w.]+)=[\d.e+-]+ \(([\d.]+)x baseline\)")


def kind_of(name: str) -> str:
    if name.startswith("contrastive"):
        return "contr"
    return "held" if name in HELD_OUT else "metric"


def rounds_of(path: Path) -> dict[int, dict[str, float]]:
    result: dict[int, dict[str, float]] = {}
    for match in ROUND.finditer(path.read_text(encoding="utf-8", errors="replace")):
        result[int(match.group(1))] = {
            name: float(ratio) for name, ratio in ENTRY.findall(match.group(2))
        }
    return result


print(
    f"{'run':34s} rd | med contr  med metric  med held | metric<1  metric>5x  max metric"
)
for argument in sys.argv[1:]:
    path = Path(argument)
    for number, ratios in sorted(rounds_of(path).items()):
        by_kind: dict[str, list[float]] = {"contr": [], "metric": [], "held": []}
        for name, ratio in ratios.items():
            by_kind[kind_of(name)].append(ratio)
        metric = by_kind["metric"]
        print(
            f"{path.stem[:34]:34s} {number:2d} | {statistics.median(by_kind['contr']):9.3f}"
            f"  {statistics.median(metric):10.2f}  {statistics.median(by_kind['held']):8.2f}"
            f" | {sum(r < 1 for r in metric):8d}  {sum(r > 5 for r in metric):9d}"
            f"  {max(metric):10.1f}"
        )
