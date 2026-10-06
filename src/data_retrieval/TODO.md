# TODO

- [x] **`dominion_parser.py` runs Selenium/network side effects at module
  import time.** Fixed 2026-10-06: the driver construction, `login()`
  calls and game-id loop now live behind a `main()` function, called only
  under `if __name__ == "__main__":`. Still a prototype (missing: type
  hints/docstrings on most functions, a `load_game` recursion where
  PRINCIPLES.md prefers a loop, a bare `except Exception: pass`) - those
  are left alone.
  (`src/data_retrieval/dominion/dominion_parser.py`)

(isotropic is now wired into `scripts/run_data_retrieval.py`
as `--source isotropic`; re-downloaded 2026-09-27: all 5 Wayback files,
~38MB, into `data/raw/isotropic/`.)
