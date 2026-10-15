# nocab_card_embedding

A personal/portfolio project to build a generic card embedding model
that represents a card from any trading card game (MTG, Pokemon,
Yu-Gi-Oh, Hearthstone, etc.) in a shared embedding space, trained via a
suite of pluggable auxiliary tasks ("dojos") rather than any single
task. The embedding model itself is the primary artifact of value —
breadth of dojos/data sources is a deliberate feature, not scope creep.

See [`src/README.md`](src/README.md) for the container map and how data
moves between containers.

## Architecture in one paragraph

Data retrieval collects raw data from external sources. The data
refinement pipeline turns that raw data into cards (the card binder),
decks (the deck box), and metrics (training inputs with true labels).
A dojo consumes a metric: it filters, transforms, splits and batches the
data. The `Trainer` (`src/training/`) feeds each batch into the model
being trained, hands the resulting embeddings back to the dojo, and the
dojo scores them, usually through a shallow decoder head private to that
dojo, with its own loss function.

## Two models

- **Single-card embedding model**: one card in, one embedding out.
- **Multi-card embedding model**: a list of cards in, one embedding per
  card out, in the same order.

The single-card model can serve multi-card workflows (run it once per
card), and the multi-card model can serve single-card workflows (pass a
list of one).

The multi-card model is **ungrouped**: given any list of cards (a deck,
a draft pool, a pack of options), it returns one embedding per card,
contextualized by the rest of that list, with no built-in concept of
"groups" within it. It never learns anything like "these cards are my
deck vs. yours"; group membership is kept out of the model so it stays a
general-purpose artifact.

Plenty of tasks do have a natural group structure (pool vs. pack
options, deck vs. deck, "here are 4 decks played in a casual commander
game, predict who will win"). For those, the grouping lives entirely in
the dojo: it takes the plain per-card embeddings and applies its own
architecture for combining and tagging them with group identity
(concatenation, a learned per-group tag vector, a small comparison
network, etc.). The training signal from those grouped tasks still
flows back into the shared multi-card model through ordinary
backpropagation, without the model ever needing to know what "group"
means.

## Getting started

```
pip install -r requirements.txt -r requirements-dev.txt
./scripts/check.sh                      # format, lint, type-check, test
./scripts/check.sh --live-data          # also test the real data under data/

PYTHONPATH=. python3 scripts/run_data_retrieval.py --list
PYTHONPATH=. python3 scripts/run_card_binder_ingestion.py --list
PYTHONPATH=. python3 scripts/run_deck_box_ingestion.py --list
PYTHONPATH=. python3 scripts/run_metrics.py --list
PYTHONPATH=. python3 scripts/preflight_dojos.py         # sanity-check dojos before a run
PYTHONPATH=. python3 scripts/smoke_test_training_loop.py
```

Each `run_*` script takes `--source <name>` or `--all`. The pipeline
runs in that order: retrieval, card binder, deck box, metrics, training.
`TODO.md` holds cross-cutting open work; each container keeps its own.

## Important Folders

Tracked by git:

```
.
├── configs/       Example training run configs (YAML) for scripts/run_training.py
├── experiments/   One folder per experiment: notebook, configs, results, bench notes
├── scripts/       Command-line entry points: run_*.py pipeline stages, check.sh, reports
├── src/           Pipeline source, one container per stage (map in src/README.md)
└── tests/         pytest suite, mirroring src/
```

Untracked folders: everything in these folders is temporary and
disposable. They are deleted or archived from time to time. None is
precious, but `data/` is the most valuable, since re-downloading and
rebuilding it takes many hours.

```
.
├── archive/   Soft-deleted code from past refactors
├── data/      Landing place for downloads and metric processing
├── logs/      Log files written by src/ and scripts/
├── models/    Model checkpoints
├── plans/     Plans written by the architect skill
├── scratch/   Agent workspace for one-off scripts and outputs
└── Nocab/     The human's private workspace
```
