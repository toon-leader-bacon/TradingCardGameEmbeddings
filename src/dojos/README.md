# dojos

Houses the pluggable auxiliary training tasks used to put training
pressure on the shared embedding space — e.g. "predict which card
doesn't belong in a deck," "classify which game a card is from,"
"predict deck membership probability." The point of a dojo is never to
be won for its own sake; it exists to shape a good general-purpose
card embedding. Nothing outside this container needs to know which
dojos exist or how any one of them works internally — `training/`
drives every dojo through the same shared `Dojo` interface.

Each dojo lives in its own subdirectory as a self-contained
mini-architecture: its own data preprocessing, label derivation, and
train/test/validate splitting and batching. A dojo owns its entire
data pipeline end to end, including translating whatever raw
identifiers its own data uses into actual cards — via `CardLookup`
(`src/data_refinement/card_binder/card_lookup.py`), handed to it once
by `prepare_splits`. `training/` never performs that translation
itself, and never sees a dojo's raw data at all — only the
already-assembled `DojoBatch`es `next_batch` returns.

**Why "dojo," not "gym":** see `src/README.md`'s naming note — a dojo
is specifically a non-episodic task (one example in, one loss out,
ordinary supervised/multi-task shape). Every task built here today is
that shape; the name `Gym` is reserved, not used, for a genuinely
different (RL environment) shape if one is ever added.

## Files

- `dojo.py` — the shared `Dojo[LabelsT]` Protocol every dojo
  implements, plus its dependent types: `CardCount` (how many cards
  one example needs), `CardSet` (one example's input cards),
  `Split` (`TRAIN`/`TEST`/`VALIDATE` — see How it works),
  `DojoBatch[LabelsT]` (already-assembled cards + labels), and
  `DecoderHead` (= `torch.nn.Module`; a dojo's own private decoder,
  never called directly outside the dojo that built it — `training/`
  only ever reads its `.parameters()`).
- `losses/` — a small shared library of reusable `Loss` Strategy
  objects a dojo can inject into its own constructor instead of
  hand-rolling loss math. See `losses/README.md`.
- `game_classification/` — `GameClassificationDojo`, the first
  concrete dojo: predicts a card's `GameId` from its own embedding.
  See `game_classification/README.md`.
- `seventeenlands/` — `MetricRegressionDojo`, a second concrete dojo:
  regresses a card's embedding against a numeric per-card 17lands
  metric, read from that metric's own `MetricResult` parquet output.
  Organized into `draft_data_dojo/`/`game_data_dojo/`/`replay_data_dojo/`
  subdirectories, one per source `data_refinement/seventeenlands/`
  pipeline, each holding thin per-metric `MetricRegressionDojo`
  subclasses. See `seventeenlands/README.md`.

## How it works

A dojo exposes six methods, called only by `training/`:

- `card_count()` — this dojo's `CardCount`, so a `Trainer` can check
  compatibility before driving it.
- `prepare_splits(corpus, held_out_cards)` — one-time setup: build and
  privately cache this dojo's train/test/validate example pools from
  `corpus`. `held_out_cards` is a small, project-curated set of
  `nocab_uuid`s a dojo may optionally exclude from its train pool
  entirely (to test generalization to cards never gradient-updated on)
  — `GameClassificationDojo` honors it; a dojo is free not to.
- `split_size(split)` / `next_batch(split, batch_size)` — draw a
  `DojoBatch`. `Split.TRAIN` samples indefinitely, with replacement,
  no exhaustiveness guarantee. `Split.TEST`/`Split.VALIDATE` are each
  a stateful cursor that covers its whole pool exactly once per full
  pass (no repeats, no gaps, a short final batch rather than padding),
  reshuffling and restarting only once a pass completes.
- `build_decoder_head(embedding_dim)` — construct this dojo's own
  decoder, sized for the encoder's embedding width.
- `compute_loss(embeddings, labels)` — run the decoder head and this
  dojo's injected `Loss` internally, return one scalar. Both stay
  entirely private to the dojo; `training/` never touches either
  directly.

## How to run

```python
from pathlib import Path

from src.data_refinement.card_binder.card_binder import CardBinder
from src.dojos.dojo import Split
from src.dojos.game_classification.game_classification_dojo import GameClassificationDojo
from src.dojos.losses.cross_entropy_loss import CrossEntropyLoss

binder = CardBinder.load([Path("data/final/cards/mtg.jsonl")])

dojo = GameClassificationDojo(
    train_ratio=0.8,
    validate_ratio=0.1,
    loss=CrossEntropyLoss(label_smoothing=0.0),
    rng_seed=0,
)
dojo.prepare_splits(binder, held_out_cards=set())

batch = dojo.next_batch(Split.TRAIN, batch_size=32)
# batch.card_sets / batch.labels are ready for an encoder + this
# dojo's own compute_loss — see src/training/README.md for the full
# loop that drives this end to end.
```
