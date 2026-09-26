import importlib.util
from pathlib import Path
from types import ModuleType

_SCRIPT = Path("scripts/report_metric_dojo_inventory.py")


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("inventory_script", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_committed_inventory_matches_the_code() -> None:
    # docs/metric_dojo_inventory.csv is generated; a new or renamed metric
    # or dojo must regenerate it in the same change.
    script = _load_script()
    expected = script._as_csv(script.inventory_rows())
    assert Path("docs/metric_dojo_inventory.csv").read_text() == expected, (
        "docs/metric_dojo_inventory.csv is stale - run "
        "`PYTHONPATH=. python3 scripts/report_metric_dojo_inventory.py`"
    )


def test_same_named_metrics_in_two_games_pair_with_their_own_dojos() -> None:
    rows = _load_script().inventory_rows()
    set_mask = {
        (row["metric_module"], row["dojo_class"])
        for row in rows
        if row["metric_class"] == "SetMaskMetric"
    }
    assert set_mask == {
        ("src.data_refinement.metrics.dominiontabs.set_mask_metric", "SetMaskDojo"),
        ("src.data_refinement.metrics.gwent_one.set_mask_metric", "SetMaskDojo"),
    }
