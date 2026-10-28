# data_refinement

Turns raw data written by `data_retrieval` (`data/raw/`) into this
project's own schema (`src/schema/`) and training labels. Everything it
writes lands under `data/final/` (cards, decks), `data/metrics/`
(labels) or `data/derived/` (training snapshots).

[`CURRENT_STATUS.md`](CURRENT_STATUS.md) tracks which retrieval
sources have binder, deck box and metric coverage, and what to build
next.

## Containers

- **[`card_binder/`](card_binder/README.md)** - `CardBinder`, the
  multi-game card store that gives every card its stable `nocab_uuid`,
  plus one `CardIngestionStage` per card source. Every other flow
  depends on it. Output: `data/final/cards/<game>.jsonl`.
- **[`deck_box/`](deck_box/README.md)** - `DeckBox`, the multi-game deck
  store (SQLite, CRUD by `nocab_uuid`), plus one `DeckExtractionStage`
  per deck source. Output: `data/final/decks/<game>.db`.
- **[`metrics/`](metrics/README.md)** - one directory per raw source,
  each turning that source into (training input, label) parquet files
  a dojo reads. Output: `data/metrics/<source>/`.
- **[`snapshots/`](snapshots/README.md)** - level 2: read-only Arrow
  card and deck snapshots compiled from the binder and deck boxes,
  stamped with the versions they came from, for fast training reads.
  Output: `data/derived/{card,deck}_snapshots/<game>/`.
- **[`seventeenlands/`](seventeenlands/README.md)** - the 17lands CSV
  chunk parser and deck identity, shared by `deck_box`'s 17lands
  extraction stage and the 17lands metrics, so both store and reference
  the same decks. Imports neither.

[`deck_ids.py`](deck_ids.py) is the one file living directly here
rather than in a container above: `deck_uuid_from_cards()`, the
content-addressed deck-id hashing both `deck_box` (a source's
extraction stage) and `metrics` (that same source's metrics) need to
agree on, so a deck gets the same id on both sides. Kept a sibling of
both rather than inside either, so neither depends on the other just
to reach it.

## How it works

```mermaid
flowchart LR
    Raw[data/raw/] --> Ingest[CardIngestionStage] --> Binder[(CardBinder)]
    Raw --> Extract[DeckExtractionStage] --> Box[(DeckBox)]
    Binder --> Extract
    Raw --> Metric[metric scanner] --> Parquet[data/metrics/*.parquet]
    Binder --> Metric
    Box --> Metric
```

Card ingestion runs first: deck extraction and metrics both look cards
up in the binder, so a game's binder must exist before either runs.

## Project data flow

The whole path from external source to training, with the file format
at each stop. Data refinement has two levels:

- **Level 1, the working stores:** the card binder (JSONL), the deck
  boxes (SQLite) and metric parquets. Ingestion writes them with
  create/update, deduplication and aliases; they are the single source
  of truth.
- **Level 2, the training snapshots:** read-only Arrow files compiled
  from level 1, stamped with the versions they came from, and rebuilt
  when those change. Training reads them as arrays.

Training is drawn as one box; its own READMEs cover the inside.

```mermaid
flowchart TD
    subgraph EXT["External sources"]
        Sources["Scryfall, 17lands, HearthstoneJSON, pokemontcg.io,<br/>fabtcg.com, gwent.one, playgwent, Dominion tabs,<br/>Spire Codex, sts.gg, ..."]
    end

    subgraph RET["data_retrieval"]
        Downloaders["one downloader per source"]
        Raw[("data/raw/&lt;source&gt;/<br/>JSON · CSV · HTML · JSONL.gz<br/>untouched dumps")]
        Downloaders --> Raw
    end
    Sources --> Downloaders

    subgraph L1["data_refinement · level 1: working stores (read-write, source of truth)"]
        Ingest["CardIngestionStage<br/>(one per card source)"]
        Binder[("CardBinder<br/>data/final/cards/&lt;game&gt;.jsonl<br/>JSONL · nocab_uuid per card")]
        Extract["DeckExtractionStage<br/>(one per deck source)"]
        Box[("DeckBox<br/>data/final/decks/&lt;game&gt;.db<br/>SQLite · decks + deck_cards")]
        Scan["metric scanners<br/>(one directory per source)"]
        Metrics[("data/metrics/&lt;source&gt;/*.parquet<br/>Parquet · (input, label) rows")]
        MetricBoxes[("metric-private deck boxes<br/>data/metrics/&lt;source&gt;/deck_box.db<br/>SQLite · sts_gg, isotropic, 17lands replay")]
        Ingest --> Binder
        Binder -->|card lookups| Extract
        Extract --> Box
        Binder --> Scan
        Box --> Scan
        Scan --> Metrics
        Scan --> MetricBoxes
    end
    Raw --> Ingest
    Raw --> Extract
    Raw --> Scan

    subgraph L2["data_refinement · level 2: training snapshots (read-only, rebuilt from level 1)"]
        CardSnap[("card snapshot<br/>data/derived/card_snapshots/&lt;game&gt;/cards.arrow<br/>Arrow · row = card_index")]
        DeckSnap[("deck snapshot<br/>data/derived/deck_snapshots/&lt;game&gt;/decks.arrow<br/>Arrow · CSR card_index + count")]
        CardSnap -->|indices| DeckSnap
    end
    Binder -->|build_training_snapshots.py| CardSnap
    Box -->|build_training_snapshots.py| DeckSnap

    subgraph TRAIN["training (abstracted)"]
        Shelf["CardShelf<br/>binders + snapshots per game,<br/>deck boxes per path, split indices"]
        subgraph CACHES["per-run split caches (built at dojo construction, reused)"]
            Splits[("metric splits<br/>data/splits/&lt;dojo&gt;_{train,test,validation}.parquet<br/>Parquet · shuffled 8/1/1")]
            DeckSplits[("contrastive deck splits<br/>data/splits/contrastive/<br/>Arrow per game + deck split key")]
        end
        Dojos["dojos<br/>build batches, own heads + losses"]
        Model["encoder_model<br/>frozen ModernBERT features (feature cache)<br/>+ trainable head"]
        Shelf --> Dojos
        Shelf --> DeckSplits
        Splits --> Dojos
        DeckSplits --> Dojos
        Dojos -->|batch of cards| Model
        Model -->|embeddings| Dojos
    end
    Metrics -->|make_splits| Splits
    Binder -->|GenericCards in memory| Shelf
    Box -->|deck dojos + snapshot stamp check| Shelf
    MetricBoxes -->|deck dojos| Shelf
    CardSnap --> Shelf
    DeckSnap --> Shelf
```

Metric splits and contrastive deck splits are per-run caches the dojos
build at construction (they depend on the run's holdout), not level 2.

## How to run

```
PYTHONPATH=. python3 scripts/run_card_binder_ingestion.py --source scryfall
PYTHONPATH=. python3 scripts/run_deck_box_ingestion.py --source play_gwent
PYTHONPATH=. python3 scripts/run_metrics.py --source gwent_one
PYTHONPATH=. python3 scripts/build_training_snapshots.py --game gwent
```

The first three take `--list`, `--source <name>` or `--all`;
`build_training_snapshots.py` takes `--game <game>` or `--all`, and runs
after the binder and deck box it compiles.
