# training

Orchestrates one `encoder_model` and one `Dojo` through an actual
training run: draws a batch from the dojo, runs its cards through the
encoder, hands the resulting embeddings to that same dojo's
`compute_loss`, and takes an optimizer step. This is the container
where the project's core idea actually happens — a dojo's
task-specific signal becoming pressure on the shared, general-purpose
embedding via backpropagation through the encoder. `training/` never
touches card lookup, id translation, or label derivation itself —
that's all already done by the time a dojo's `next_batch` returns (see
`src/dojos/README.md`).

`training/` is a flat directory: a `Trainer` ABC plus concrete
implementations as sibling files, one per model-arity/regime
combination — `SingleCardTrainer` is the first. Each `Trainer`
implementation's `__init__` always receives an already-constructed
`Dojo`; none constructs one itself, which is what lets a `Trainer`
drive any dojo without needing dojo-specific knowledge.

## Files

- `trainer.py` — the shared `Trainer` ABC (`train(num_steps,
  batch_size) -> EvaluationResult`), and `EvaluationResult`
  (`test_loss`, `validate_loss`).
- `single_card_trainer.py` — `SingleCardTrainer`, the first `Trainer`
  implementation. Only accepts dojos where `card_count() ==
  CardCount(1, 1)`; registers both the encoder's and the dojo's
  decoder head's parameters with one shared optimizer at construction.

## How it works

`SingleCardTrainer.train(num_steps, batch_size)` runs three sequential
phases, never interleaved:

1. **Train** — `num_steps` gradient steps, each drawing one batch via
   `dojo.next_batch(Split.TRAIN, batch_size)`, embedding it, computing
   loss via `dojo.compute_loss`, and stepping the optimizer. The only
   phase that calls `.backward()`.
2. **Test** — one full no-gradient pass over `Split.TEST`
   (`math.ceil(dojo.split_size(Split.TEST) / batch_size)` batches),
   reported as `EvaluationResult.test_loss`. Currently just reported,
   not acted on.
3. **Validate** — one full no-gradient pass over `Split.VALIDATE`,
   reported as `EvaluationResult.validate_loss` — this run's actual
   result, on data untouched by phases 1 or 2.

## How to run

```python
from pathlib import Path

from src.data_refinement.card_binder.card_binder import CardBinder
from src.dojos.game_classification.game_classification_dojo import GameClassificationDojo
from src.dojos.losses.cross_entropy_loss import CrossEntropyLoss
from src.encoder_model.single_card.toy_single_card_model import ToySingleCardModel
from src.training.single_card_trainer import SingleCardTrainer

binder = CardBinder.load([Path("data/final/cards/mtg.jsonl")])

trainer = SingleCardTrainer(
    model=ToySingleCardModel(embedding_dim=32, vocab_size=4096),
    dojo=GameClassificationDojo(
        train_ratio=0.8,
        validate_ratio=0.1,
        loss=CrossEntropyLoss(label_smoothing=0.0),
        rng_seed=0,
    ),
    corpus=binder,
    held_out_cards=set(),
    learning_rate=1e-3,
)
result = trainer.train(num_steps=1000, batch_size=64)
print(result.test_loss, result.validate_loss)
```
