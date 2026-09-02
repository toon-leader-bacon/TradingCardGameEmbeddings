# nocab_card_embedding

## Quick Start Overview

The goal is to build an effective and general embedding model for a Trading Card Game, for any arbitrary TCG. This is a personal hobby project, so there are a lot of options for scalability here too.
I've gotten a rough outline of the high level architecture and it is as follows:
Data retrieval is responsible for collecting raw data from external sources.
This raw data is 'refined' via the Data Refinement pipline, either to extract new cards (to add to our binder) or to create metrics (training data and true labels)
Metric data is consumed by a dojo, which is responsible for filtering, transforming, splitting and batching the data.
The dojo hands each batch to a core TrainingLoop implementation, which is responsible for coordinating the dojo data feed, into the model being trained to generate embeddings.
Those embeddings are then handed back to the dojo for evaluation and loss calculation (most likely via a shallow decoder head private/ custom to the dojo-metric and evaluated with a loss function there)

There are two types of models I care about to build:

- Singe card embedding model (single card in, single embedding out)
- Multi card embedding model (multiple cards in, one embedding per card out in the same order)

The single card embedding model can support multi-card workflows (via running it multiple times)
The multi card embedding model can support single-card workflows (via passing a single card)

Additionally, there is a concept of multi-group tasks like "Here are 4 decks played in a casual commander game, predict who will win".
This can be modeled as multiple multi-card embedding tasks, that get merged/ consumed by a more complex decoder head on top of it.

## README.md

A personal/portfolio project to build a generic card embedding model
that represents a card from any trading card game (MTG, Pokemon,
Yu-Gi-Oh, Hearthstone, etc.) in a shared embedding space, trained via a
suite of pluggable auxiliary tasks ("dojos") rather than any single
task. The embedding model itself is the primary artifact of value —
breadth of dojos/data sources is a deliberate feature, not scope creep.

See [`src/README.md`](src/README.md) for the full container map. This
file is just the entry point: what the project is, and how data moves
through it.

## The two data flows

Everything in this project reduces to two flows, both handing data off
between containers as files on disk rather than in-memory objects:

1. **Card ingestion** — raw data → a `CardIngestionStage` → a
   `CardBinder`, saved to `data/final/cards/<game>.jsonl`. This is the
   one thing every other flow depends on: it's how a raw card, from
   any source, gets this project's own stable identity
   (`nocab_uuid`).
2. **Metric → dojo** — raw data → a metric scanner → a parquet file of
   per-card metric values → a `Dojo`, which reads that parquet file as
   its label source and the `CardBinder` file (flow 1's output) to
   resolve each label back to an actual card. [`docs/metric_dojo_inventory.csv`](docs/metric_dojo_inventory.csv)
   lists every metric ↔ dojo pairing that exists today.

`training/` drives one encoder model against one (or more) `Dojo`s,
never touching either flow's raw data directly — see
[`src/README.md`](src/README.md#how-the-pieces-fit-together) for the
precise class names and [`src/training/README.md`](src/training/README.md)
for how a training run is actually assembled.

## Multi-card embedding model

Beyond the single-card model, this project plans to train and offer
one **ungrouped** multi-card embedding model: given a list of cards
(a deck, a draft pool, a pack of options — any set, any size), it
returns one embedding per input card, contextualized by the rest of
that list, with no built-in concept of "groups" within that list. It
never learns anything like "these cards are my deck vs. yours" —
group membership is deliberately kept out of the model itself so it
stays a general-purpose artifact usable for any task, not one shaped
around a particular grouped comparison.

We still train on plenty of tasks that *do* have a natural group
structure — pool vs. pack options, deck vs. deck, and similar. For
those, the grouping is handled entirely downstream, inside the dojo:
a dojo takes the plain per-card embeddings this model already
produces and applies its own architecture for combining and tagging
them with group identity (concatenation, a learned per-group tag
vector, a small comparison network, etc.). Different dojos are free
to demonstrate different consumption patterns for the same underlying
embeddings — the training signal from those grouped tasks still flows
back into the shared multi-card model through ordinary
backpropagation, without the model itself ever needing to know what
"group" means.
