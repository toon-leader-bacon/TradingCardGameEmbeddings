import itertools, os
from pathlib import Path
os.environ["NOCAB_SPLIT_DIRECTORY"] = "scratch/cardreward/splits2"
import pyarrow.parquet as pq
import torch
from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.metrics.sts2_runs.card_removal_pick_metric import CardRemovalPickMetric
from src.data_refinement.metrics.sts2_runs.card_reward_pick_metric import CardRewardPickMetric
from src.data_refinement.metrics.sts2_runs.card_upgrade_pick_metric import CardUpgradePickMetric
from src.data_refinement.metrics.sts2_runs.shop_purchase_pick_metric import ShopPurchasePickMetric
from src.data_refinement.metrics.sts2_runs.run_parser import Sts2RunParser
from src.data_refinement.metrics.sts2_runs.scanner import default_run_sources, scored_run, RunExclusion
from src.dojos.dojo import BatchBudget
from src.dojos.sts2_runs.card_removal_pick_dojo import CardRemovalPickDojo
from src.dojos.sts2_runs.card_upgrade_pick_dojo import CardUpgradePickDojo
from src.dojos.sts2_runs.shop_purchase_pick_dojo import ShopPurchasePickDojo
from src.schema.game_id import GameId
from src.schema.holdout import HoldoutSpec
from src.schema.splits import Split

b = CardBinder.load([CardBinder.default_output_path(GameId.SLAY_THE_SPIRE_2)])
out = Path("scratch/cardreward/p2")
out.mkdir(parents=True, exist_ok=True)
metrics = {
    "reward": CardRewardPickMetric(b, out / "reward.parquet"),
    "shop": ShopPurchasePickMetric(b, out / "shop.parquet"),
    "removal": CardRemovalPickMetric(b, out / "removal.parquet"),
    "upgrade": CardUpgradePickMetric(b, out / "upgrade.parquet"),
}
src = default_run_sources()[0]
parser = Sts2RunParser(b, src.stage)
n = 0
raw_upgrade_events = raw_removals = 0
for row in itertools.islice(src.stage.runs(src.stage.raw_files(src.raw_path)[0]), 3000):
    v = scored_run(parser.parse(row))
    if isinstance(v, RunExclusion):
        continue
    n += 1
    for m in metrics.values():
        m.accumulate(v)
for m in metrics.values():
    m.finalize()
print("runs", n)
for name in metrics:
    t = pq.read_table(out / f"{name}.parquet")
    print(name, "rows", t.num_rows)
dojos = {
    "shop": ShopPurchasePickDojo,
    "removal": CardRemovalPickDojo,
    "upgrade": CardUpgradePickDojo,
}
for name, cls in dojos.items():
    dojo = cls(b, HoldoutSpec.no_holdout(), 8, path_to_training_data=out / f"{name}.parquet", rng_seed=0)
    counts = {s.name: dojo.example_count(s) for s in (Split.TRAIN, Split.TEST, Split.VALIDATION)}
    batch = next(iter(dojo.batches(Split.TRAIN, BatchBudget(max_cost=4000, cost_of=lambda c: 1))))
    emb = [[[torch.randn(8) for _ in g] for g in ex] for ex in batch.inputs]
    loss = dojo.compute_loss(emb, batch)
    sizes = [len(ex[0]) for ex in batch.inputs]
    print(name, counts, "avg options", sum(sizes) / len(sizes), "loss", float(loss), "baseline", float(dojo.baseline_loss(batch)))
