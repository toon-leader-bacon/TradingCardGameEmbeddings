# tests

pytest suite, mirroring `src/`'s layout (one directory per container)
plus `scripts/` for tests of the scripts themselves. `conftest.py`
patches `time.sleep` to a no-op for every test so retry/backoff paths
run instantly.

Tests never need `data/`: they build small binders, deck boxes and
parquet files in `tmp_path`. The one test that checks live project data
skips when that data is absent.

Run everything (format, lint, type-check, tests) with
`./scripts/check.sh`, or just the tests with `python3 -m pytest tests -q`.
