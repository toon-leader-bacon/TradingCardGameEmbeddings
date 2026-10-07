# TODO

- Make a JSONL file manager (`file_manager_jsonl.py`) if a metric ever
  writes JSONL instead of parquet.
- **Delete the un-prefixed leftovers in `data/splits/`.** Dojo tests
  used to write placeholder splits there (fixed: `tests/conftest.py`
  points `DojoConfig`'s default split directory at a per-test temp
  directory through `$NOCAB_SPLIT_DIRECTORY`). The old files, such as
  `held_out_card_gwent_*`, `isotropic_source_*` and the pre-catalog
  `color_mask_*` and `kingdom_game_length_*`, are read by no catalog key
  (those are prefixed with the key) but can hide a split change.
