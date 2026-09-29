# recording

What a run leaves behind: values describing each round, files on disk, and
hooks for observers.

## Files

- [reports.py](reports.py): `RoundReport`, `CheckpointRecord`,
  `TrainingResult`, `DojoStatus`, and `mean_test_loss_over`, the score
  used to pick a phase's best checkpoint.
- [checkpointer.py](checkpointer.py): `Checkpointer` Protocol and
  `DirectoryCheckpointer`, which writes a weights snapshot (`state.pt`), an
  encoder-only file (`encoder.pt`, the seam for Hugging Face export), the
  run's `HoldoutSpec` (`holdout.json`, lossless JSON) and a manifest. The
  same module reads a checkpoint back: `load_encoder_weights` (strict load
  of `encoder.pt` into a caller-built model) and `load_checkpoint_holdout`
  (`FileNotFoundError` for checkpoints written before `holdout.json`).
- [run_listener.py](run_listener.py): `RunListener` Observer Protocol,
  `LoggingRunListener`, and `CsvRunListener` (appends per-round, per-dojo
  losses and checkpoint records to two CSV files, for tailing or loading
  with pandas during an overnight run). `RoundRow` is the single
  definition of the rounds-CSV format: the header is its field names, and
  `read_rounds_csv` parses a file back into `RoundRow`s. The listener
  creates directories on first write (constructing one touches nothing),
  and refuses (`ValueError`) to append to a CSV whose header differs, e.g.
  one written before a column was added.

## How it works

After each round `Trainer` builds a `RoundReport`, notifies every listener,
and, if the report is the phase's best, asks the checkpointer to save.
