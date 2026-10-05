# TODO

- [ ] **`CardVaultFabtcgCardIngestionStage.ingest()` duplicates a scan
  loop.** Two near-identical ~18-line blocks (open the raw file in
  binary, wrap in a `tqdm` progress bar sized by byte offset, iterate
  `csv.DictReader`, track `raw_bytes.tell()`) differ only in the row
  filter and which per-row method is called, giving the method 3-level
  nesting (`with`/`with`/`for`/`if`) across ~95 lines. A
  `_scan_csv_rows_with_progress(raw_path, desc) -> Iterator[(row,
  dict)]` helper would remove the duplicate loop and flatten the
  nesting (PRINCIPLES.md §2 duplication, §4 size/nesting). Flagged
  during a 2026-10-05 project-wide cleanup scan, not yet fixed.
  (`src/data_refinement/card_binder/cardvault_fabtcg/ingestion_stage.py`,
  lines 135-231)
