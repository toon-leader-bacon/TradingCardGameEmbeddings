"""Paired difference between two evaluated encoders (Method, step 4).

Delta = B - A in diet-weighted mean same-deck retrieval AUC, over the
same fixed VALIDATION decks (evaluate.py's per_deck.csv for each label).
Per game, the mean over decks of (auc_B - auc_A); the weighted mean over
the 15 trained games with evaluate.py's DIET_WEIGHTS. The 95% CI is a
paired bootstrap that resamples decks within each game (stratified, so
every game keeps its deck count). Also prints the linear-cls difference
from results/eval_summary.csv and the quality rule from the Aim.

Appends one row per comparison to results/compare.csv (a re-run of the
same pair replaces its row) and writes per-game deltas to
results/compare_by_game/<A>__<B>.csv.

Run (project root, PYTHONPATH=.):
    python experiments/2026-11-11_teacher_vs_real_pairs/analysis/compare.py <label A> <label B>
"""

import csv
import sys
from pathlib import Path

import numpy as np

EXPERIMENT = "2026-11-11_teacher_vs_real_pairs"
INTRINSIC = Path("data/evaluations") / EXPERIMENT / "intrinsic"
RESULTS = Path("experiments") / EXPERIMENT / "results"
# Same table as evaluate.py's DIET_WEIGHTS, keyed by GameId value.
DIET_WEIGHTS = {
    "mtg": 160, "yugioh": 106, "pokemon": 78, "hearthstone": 66, "flesh_and_blood": 62,
    "digimon": 54, "one_piece": 40, "slay_the_spire_2": 21, "star_wars_unlimited": 20,
    "dragon_ball_masters": 19, "gundam": 19, "dragon_ball_fusion_world": 18,
    "lorcana": 16, "battle_spirits_saga": 16, "dominion": 12,
}
BOOTSTRAP = 2000
SEED = 0
MARGIN = 0.02


def per_deck(label: str) -> dict[str, tuple[str, float]]:
    """deck uuid -> (game, auc) for one encoder."""
    path = INTRINSIC / label / "deck_retrieval_auc" / "per_deck.csv"
    with path.open(encoding="utf-8", newline="") as f:
        return {r["deck_uuid"]: (r["source_game"], float(r["auc"])) for r in csv.DictReader(f)}


def paired_by_game(a: dict, b: dict) -> dict[str, np.ndarray]:
    """game -> per-deck (auc_B - auc_A) over decks both encoders scored."""
    shared = sorted(a.keys() & b.keys())
    if len(shared) < len(a.keys() | b.keys()):
        print(f"warning: {len(a.keys() ^ b.keys())} decks scored by only one encoder; dropped")
    diffs: dict[str, list[float]] = {}
    for uuid in shared:
        diffs.setdefault(a[uuid][0], []).append(b[uuid][1] - a[uuid][1])
    return {g: np.asarray(d) for g, d in diffs.items()}


def weighted(means: dict[str, float]) -> float:
    total = sum(DIET_WEIGHTS[g] for g in means if g in DIET_WEIGHTS)
    return sum(DIET_WEIGHTS[g] * m for g, m in means.items() if g in DIET_WEIGHTS) / total


def bootstrap_ci(diffs: dict[str, np.ndarray]) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    trained = {g: d for g, d in diffs.items() if g in DIET_WEIGHTS}
    draws = np.empty(BOOTSTRAP)
    for i in range(BOOTSTRAP):
        draws[i] = weighted({g: d[rng.integers(0, len(d), len(d))].mean() for g, d in trained.items()})
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def linear_cls(label: str) -> float | None:
    with (RESULTS / "eval_summary.csv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["encoder"] == label and row["linear_cls"]:
                return float(row["linear_cls"])
    return None


def quality_rule(delta: float, lo: float, hi: float, cls_delta: float | None) -> str:
    """The Aim's quality verdict for B (T) against A (R)."""
    cls_drop = cls_delta is not None and cls_delta < -MARGIN
    if delta <= -MARGIN and hi < 0 or cls_drop:
        return "refuted"
    if delta >= MARGIN and lo > 0:
        return "supported"
    return "no clear difference"


def write_row(path: Path, row: dict, keys: tuple[str, ...]) -> None:
    rows = []
    if path.exists():
        with path.open(encoding="utf-8", newline="") as f:
            rows = [r for r in csv.DictReader(f) if tuple(r[k] for k in keys) != tuple(str(row[k]) for k in keys)]
    rows.append(row)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    label_a, label_b = sys.argv[1], sys.argv[2]
    diffs = paired_by_game(per_deck(label_a), per_deck(label_b))
    means = {g: float(d.mean()) for g, d in diffs.items()}
    delta = weighted(means)
    lo, hi = bootstrap_ci(diffs)
    cls_a, cls_b = linear_cls(label_a), linear_cls(label_b)
    cls_delta = cls_b - cls_a if cls_a is not None and cls_b is not None else None
    verdict = quality_rule(delta, lo, hi, cls_delta)
    row = {
        "a": label_a, "b": label_b, "delta_weighted_auc": delta, "lo95": lo, "hi95": hi,
        "n_decks": sum(len(d) for d in diffs.values()),
        "linear_cls_a": cls_a, "linear_cls_b": cls_b, "linear_cls_delta": cls_delta,
        "quality_rule": verdict,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    write_row(RESULTS / "compare.csv", row, ("a", "b"))
    by_game = RESULTS / "compare_by_game"
    by_game.mkdir(exist_ok=True)
    with (by_game / f"{label_a}__{label_b}.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("game", "n_decks", "delta_auc", "diet_weight"))
        for g in sorted(diffs, key=lambda g: -DIET_WEIGHTS.get(g, 0)):
            writer.writerow((g, len(diffs[g]), means[g], DIET_WEIGHTS.get(g, "")))
    print(f"{label_b} - {label_a}: weighted deck AUC {delta:+.4f} [{lo:+.4f}, {hi:+.4f}], "
          f"linear-cls delta {cls_delta if cls_delta is None else f'{cls_delta:+.4f}'}; quality rule: {verdict}")
    for g in sorted(diffs, key=lambda g: -DIET_WEIGHTS.get(g, 0)):
        print(f"  {g:26s} n={len(diffs[g]):4d}  {means[g]:+.4f}")


if __name__ == "__main__":
    main()
