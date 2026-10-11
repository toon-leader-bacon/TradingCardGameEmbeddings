"""Per-round summary of a run's rounds.csv: timing and normalized-loss trend."""

import sys

import pandas as pd

d = pd.read_csv(sys.argv[1])
d["held"] = d.status.isna()
d["kind"] = d.dojo.str.startswith("contrastive").map({True: "contrastive", False: "metric"})
d.loc[d.held, "kind"] = "held_out"
rounds = d.groupby("round_index").elapsed_seconds.first()
print("round end (min):", (rounds / 60).round(1).to_dict())
print("round durations (min):", (rounds.diff().fillna(rounds.iloc[0]) / 60).round(1).to_dict())
table = d.pivot_table(index="round_index", columns="kind", values="normalized_test_loss", aggfunc="median")
table["trained_below_1"] = d[~d.held].groupby("round_index").normalized_test_loss.apply(lambda s: (s < 1).sum())
print(table.round(3).to_string())
last, first = d.round_index.max(), d.round_index.min()
wide = d[~d.held].pivot(index="dojo", columns="round_index", values="normalized_test_loss")
print("worst now:")
print(wide.sort_values(last, ascending=False).head(8).round(2).to_string())
if last > first:
    change = (wide[last] - wide[last - 1]).sort_values()
    print("most worsened since last round:")
    print(change.tail(5).round(2).to_string())
