# TODO

- [ ] **`dominion_parser.py` runs Selenium/network side effects at module
  import time.** It creates two Chrome drivers and calls `login()` at
  the bottom of the file, outside any `if __name__ == "__main__":`
  guard - importing this module (or its package) anywhere, e.g. from a
  test collector, launches browsers and hits the network. The container
  README calls this file "a prototype replay scraper," which explains
  the rough edges (also missing: type hints/docstrings on most
  functions, a `load_game` recursion where PRINCIPLES.md prefers a
  loop, a bare `except Exception: pass`), but the import-time side
  effect is the one worth fixing on its own: move the driver
  construction and `login()` call behind a `main()`/entry-point guard.
  Flagged during a 2026-10-05 project-wide cleanup scan, not yet fixed.
  (`src/data_retrieval/dominion/dominion_parser.py`)

(isotropic is now wired into `scripts/run_data_retrieval.py`
as `--source isotropic`; re-downloaded 2026-09-27: all 5 Wayback files,
~38MB, into `data/raw/isotropic/`.)
