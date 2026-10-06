# TODO

- [x] **`CardVaultFabtcgCardIngestionStage.ingest()` duplicates a scan
  loop.** Fixed 2026-10-06: a new module-level
  `_scan_csv_rows_with_progress(raw_path, desc) -> Iterator[dict]`
  helper holds the open-in-binary/tqdm/csv.DictReader/`raw_bytes.tell()`
  loop once; `ingest()`'s two passes now each just iterate it with their
  own row filter and per-row call, flattening the old 3-level nesting.
  (`src/data_refinement/card_binder/cardvault_fabtcg/ingestion_stage.py`)
