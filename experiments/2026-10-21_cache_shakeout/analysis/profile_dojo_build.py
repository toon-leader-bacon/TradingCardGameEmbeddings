"""cProfile the building of named dojos (CPU only), one at a time, sharing one
CardShelf as a run does. Usage (project root, ROCm venv):
    PYTHONPATH=. python experiments/2026-10-21_cache_shakeout/analysis/profile_dojo_build.py <dojo> ... > results/<file>.txt
"""
import cProfile, pstats, sys, time, logging
logging.basicConfig(level=logging.INFO)
sys.path.insert(0, ".")
from src.training.run_config import read_config_document, parse_run_config
from src.training.dojo_catalog import DOJO_CATALOG, build_dojos
from src.training.dojo_recipes.card_shelf import CardShelf
from src.training.dojo_recipes.dojo_build_context import DojoBuildContext
from src.dojos.loss.regression_objective import RegressionObjective
cfg = parse_run_config(read_config_document(__import__("pathlib").Path("experiments/2026-10-21_cache_shakeout/configs/cached.yaml")), tuple(DOJO_CATALOG))
ctx = DojoBuildContext(shelf=CardShelf(), holdout=cfg.plan.holdout, card_embedding_size=cfg.model.card_embedding_size,
    rng_seed=cfg.plan.seed, mod_overrides=cfg.mod_overrides, staple_thresholds=cfg.staple_thresholds,
    regression_objective=RegressionObjective.for_kind(cfg.regression_loss))
names = sys.argv[1:]
for name in names:
    t = time.perf_counter()
    prof = cProfile.Profile(); prof.enable()
    build_dojos([name], ctx)
    prof.disable()
    print(f"=== {name}: {time.perf_counter()-t:.1f}s")
    pstats.Stats(prof).sort_stats("cumulative").print_stats(25)
