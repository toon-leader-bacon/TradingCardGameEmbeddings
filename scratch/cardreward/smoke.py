import itertools, os, time
from pathlib import Path
os.environ["NOCAB_SPLIT_DIRECTORY"] = "scratch/cardreward/splits"
import torch
from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.metrics.sts2_runs.card_reward_pick_metric import CardRewardPickMetric
from src.data_refinement.metrics.sts2_runs.run_parser import Sts2RunParser
from src.data_refinement.metrics.sts2_runs.scanner import default_run_sources, scored_run, RunExclusion
from src.dojos.sts2_runs.card_reward_pick_dojo import CardRewardPickDojo
from src.dojos.dojo import BatchBudget
from src.schema.game_id import GameId
from src.schema.holdout import HoldoutSpec
from src.schema.splits import Split

b = CardBinder.load([CardBinder.default_output_path(GameId.SLAY_THE_SPIRE_2)])
out = Path("scratch/cardreward/card_reward_pick.parquet")
m = CardRewardPickMetric(b, out)
src = default_run_sources()[0]
parser = Sts2RunParser(b, src.stage)
for row in itertools.islice(src.stage.runs(src.stage.raw_files(src.raw_path)[0]), 3000):
    v = scored_run(parser.parse(row))
    if not isinstance(v, RunExclusion):
        m.accumulate(v)
m.finalize()
dojo = CardRewardPickDojo(b, HoldoutSpec.no_holdout(), 8, path_to_training_data=out, rng_seed=0)
print("examples", {s.name: dojo.example_count(s) for s in (Split.TRAIN, Split.TEST, Split.VALIDATION)})
batch = next(iter(dojo.batches(Split.TRAIN, BatchBudget(max_cost=4000, cost_of=lambda c: 1))))
emb = [[[torch.randn(8) for _ in g] for g in ex] for ex in batch.inputs]
loss = dojo.compute_loss(emb, batch)
skips = sum(label == len(ex[0]) for ex, label in zip(batch.inputs, batch.labels))
print("batch", len(batch.labels), "skip share", skips / len(batch.labels), "loss", float(loss), "baseline", float(dojo.baseline_loss(batch)))
