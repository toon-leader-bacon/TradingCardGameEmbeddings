# TODO

- Reorganize this top level directory to have the following subs
  - card_binder (already exists)
  - metrics
Each of these subs should have a dedicated sub-diretory per data source (as appropriate)

Additionally, I think there needs to be a concept of a few different types of refinements:

- Card extraction, already exists and is captured by the card_binder
- deck extraction: Unordered single list fo Nocab UUIDs (Does not currently exist at time of writing)
- other metric extraction: The thing that produces TrainingDatum (input, true-label) pairs (many of which may be single card inputs, or deck inputs)

However, I'm not really sure there's a super useful distinction between deck extraction and other metric extraction.
I think it's so common that it might just be worthwhile to do it (so many data sources only provide flat decks without any meta data like win-rates, draft order, deck vs deck results etc)
But, many data sources/ games also have custom details for representing a deck (In MTG the commander is important and separate from the rest of the deck for example)

So to really represent the suite of possible metrics, maybe I want both? A  MtG flat-deck extraction tool that produces the single list of Nocab UUIDs.
Every possible TCG could fit that schema (at the cost of potentially removing game specific information)
But also there may be a separate other metric extraction that does a very similar job (extracting a mtg deck from raw data source),
except it keeps the game specifc structure (like MtG commanders, or FaB heroes and Pitchs for example)

This may break the convention I set earlier that "other metrics shall return TrainingData (input, true-label)", because a game aware deck extraction is not producing true labels, just more structure decks.
However, the spirit of the project is that dojos will ingest the training data and use it as training data.
I can imagine a suite of dojos that consume MtG specific deck structures and just-in-time convert it into training data.
IE: The dojo takes a flat deck of NocabUUIDs and masks one, then asks the embedding model to guess what one is missing
IE: The dojo takes a game specific structure, masks out one important part (like the commander) and asks what card is missing
These two above examples are similar, but distinct in that the commander is a very special card to guess, and commander decks are often built around the commander so guessing the commander should be easier than guessing a random card masked from a random deck.

So I'm not 100% sure if it's a good distinction to organize the data_refinement into the 3 different refinement types (card, flat deck, other metrics), but I am ~80% sure it's a good idea so I say we go for it.
