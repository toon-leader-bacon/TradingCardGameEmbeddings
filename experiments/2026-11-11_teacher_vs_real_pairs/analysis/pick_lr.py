"""Apply the LR-pilot rule fixed in the Method (step 1) to the six pilot
runs' rounds.csv files, and measure step times for sizing the main runs.

Rule: per arm, the encoder_lr with the lowest median own-task normalized
TEST loss at the last round, over the 7 games with at least 128 teacher
TEST anchors; within 2% of the best, the larger LR wins. Also reports the
other arm's (held-out) dojos' median, and ms/step between the last two
rounds (TEST included), which excludes the first round's warm-up.

Writes results/lr_pilot.csv; prints the pick per arm.
Run (project root): python experiments/2026-11-11_teacher_vs_real_pairs/analysis/pick_lr.py
"""

import csv
import statistics
from pathlib import Path

RUNS = Path("data/runs/2026-11-11_teacher_vs_real_pairs")
RESULTS = Path("experiments/2026-11-11_teacher_vs_real_pairs/results")
LRS = ("1e-04", "3e-04", "1e-03")
BIG_GAMES = ("mtg", "yugioh", "pokemon", "hearthstone", "flesh_and_blood", "digimon", "one_piece")
PREFIX = {"R": "contrastive", "T": "teacher_contrastive"}
OTHER = {"R": "T", "T": "R"}
TIE = 0.02


def summarize(arm: str, lr: str) -> dict | None:
    path = RUNS / f"pilot_{arm}_lr{lr}" / "rounds.csv"
    if not path.exists():
        return None
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    last = max(int(r["round_index"]) for r in rows)
    final = {r["dojo"]: float(r["normalized_test_loss"]) for r in rows if int(r["round_index"]) == last}
    own = [final[f"{PREFIX[arm]}.{g}"] for g in BIG_GAMES]
    other = [final[f"{PREFIX[OTHER[arm]]}.{g}"] for g in BIG_GAMES]
    timing = sorted({(int(r["round_index"]), int(r["step"]), float(r["elapsed_seconds"])) for r in rows})
    ms_per_step = None
    if len(timing) >= 2:
        (_, s0, t0), (_, s1, t1) = timing[-2], timing[-1]
        ms_per_step = 1000 * (t1 - t0) / (s1 - s0)
    return {
        "arm": arm, "encoder_lr": lr, "last_round": last,
        "own_median": statistics.median(own), "other_median": statistics.median(other),
        "ms_per_step_last_round": ms_per_step,
    }


def main() -> None:
    rows = [r for arm in ("R", "T") for lr in LRS if (r := summarize(arm, lr))]
    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "lr_pilot.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)
    for arm in ("R", "T"):
        arm_rows = [r for r in rows if r["arm"] == arm]
        if len(arm_rows) < len(LRS):
            print(f"{arm}: {len(arm_rows)} of {len(LRS)} runs done; no pick")
            continue
        best = min(r["own_median"] for r in arm_rows)
        tied = [r for r in arm_rows if r["own_median"] <= best * (1 + TIE)]
        pick = max(tied, key=lambda r: float(r["encoder_lr"]))
        print(f"{arm}: pick encoder_lr {pick['encoder_lr']} (median {pick['own_median']:.4f}; best {best:.4f})")


if __name__ == "__main__":
    main()
