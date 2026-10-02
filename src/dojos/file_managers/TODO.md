# TODO

- Make a JSONL file manager (`file_manager_jsonl.py`) if a metric ever
  writes JSONL instead of parquet.
- **Some dojo tests write split files into the real `data/splits/`.**
  A wrapper test that builds a dojo without an `output_directory` uses
  `DojoConfig`'s relative default, so it writes placeholder splits such
  as `data/splits/held_out_card_gwent_*.parquet` and
  `isotropic_source_*.parquet`. Later runs then reuse those files
  silently. No catalog key reads them (catalog splits are prefixed with
  the key, e.g. `final_decks.held_out_card_gwent_*`), but they can hide
  a split change from the test that wrote them. Fix: an autouse fixture
  that chdirs dojo tests into `tmp_path`, or a split directory taken
  from an environment variable. Then delete the un-prefixed leftovers in
  `data/splits/` (they also include pre-catalog splits from 09-29/30,
  e.g. `color_mask_*`, `kingdom_game_length_*`).
