# recording

What a run leaves behind: values describing each round, files on disk, and
hooks for observers.

## Files

- [reports.py](reports.py): `RoundReport`, `CheckpointRecord`,
  `TrainingResult`, `DojoStatus`, and `mean_test_loss_over`, the score
  used to pick a phase's best checkpoint.
- [checkpointer.py](checkpointer.py): `Checkpointer` Protocol and
  `DirectoryCheckpointer`, which writes a weights snapshot, an
  encoder-only file (the seam for Hugging Face export) and a manifest.
- [run_listener.py](run_listener.py): `RunListener` Observer Protocol and a
  logging listener.

## How it works

After each round `Trainer` builds a `RoundReport`, notifies every listener,
and, if the report is the phase's best, asks the checkpointer to save.
