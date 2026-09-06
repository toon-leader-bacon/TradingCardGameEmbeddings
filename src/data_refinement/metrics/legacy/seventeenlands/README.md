# seventeenlands

All the logic associated with taking raw 17 lands data and converting them into metrics that can be used with a dojo to train a model.

## Raw Data

All raw data comes in the form of CSV files.
The raw data is organized as follows: For each Expansion & format combo, there are up to 3 different types of data:

- Draft data
- Game Data
- Replay Data

### Draft Data

Each row in this CSV is one draft pick decision. The "pick" is the name of the card selected, and the Pool columns are the cards that are already in the player's pool (previously picked) and the Pack columns are the cards that are avaliable to be drafted.

### Game Data

Each row is the metada associated with a game. Always from the current player's perspective, so you can see the cards in the player's deck and the result (win loss) of that game.
For more rich game data, see Replay Data.

### Replay Data

Each row is a full game (up to 30 turns). The data is always from the current player's perspective.

Opponent data is hidden or obfuscated. For example, the column `user_turn_1_eot_user_cards_in_hand` represents "At the end of turn for 'user' these are the arena IDs of all the cards in 'user's hand", which is a list of ID numbers.
However, the similar column `user_turn_1_eot_oppo_cards_in_hand` represents "At the end of turn for 'user', the opponent has this many cards in hand" which is a single number. The exact card IDs are private to the opponent, so we can't see it in this data.

However, public data is visible, so a column like `user_turn_1_eot_oppo_creatures_in_play` /is/ a list of Arena IDs, because even tho it's referencing what creatures the opponent has at end of turn, they are in play and therefore are public so we can get those exact Arena IDs.

When in doubt, double check the actual data by carefully loading a small chunk into a python pandas data frame and querying and comparing a few results.

## Metrics

There are two main types of metric builders:

- Accumulation metric builders
- Streaming metric builders

Further multiplexing this, there are two main types of metrics:

- Single card metrics
- Multi card metrics

### Accumulation metric builders

These metric builders can only produce their output data once all the data has been processed.
For example, to compute the average win rate of a card, we need to tally the total number of wins and the total number of instances of the card in a deck.
Therefore, the metric builders "accumulate" metadata about that which they care about and only once "finalize" is called will they compute the final metric and write to a file.

### Streaming metric builders

These metric builders can produce their output metric with just a single row of raw data so do not need to maintain internal state.
For example, determining which card was drafted given a collection of options. Each row in the raw data already features this data, so the streaming metric typically will filter, reduce and simplify raw data into something more managable for dowstream consumption
The finalize function for streaming metrics are typically a no-op because each accumulate step can write the full metric out already.

### Single vs multi card metrics

Metrics are just labeled data mapping inputs to outputs.
Single card metrics are those which have just one card as the input. For example:

- Predicting average win rate of a card
- Predicting average draft number of a card (first pick cards are typically more powerful than last pick cards)
- Predicting the average win % if the card is in your opening hand
- Predicting what % of decks sideboard the card
- etc.

Multi card metrics are cards which have multiple cards as input. For example:

- Predict if a given deck wins or looses the game it's in
- Given a set of draftable cards predict which one specifically was drafted

Multi card metrics may be more complex, and have more than one 'group' of multiple cards.
For example, you can more accurately predict what card will be drafted if you consider:

- What cards has the player already drafted (in the pool)
- What cards are avaliable to be drafted (in the pack)

This lends itself to be a multi-card metric, with two groups of inputs (pool and pack) with a single output (card actually drafted)

How the multi-card or multi-group metrics are consumed is not relevant for creating metrics, and will be considered and managed by the relevant dojo that consumes this metric for training.

## Scanner

A scanner is the main loop tha processes the raw data file, splits it up into chunks and feeds those chunks into a collection of metrics.
As of writing, we only have a CSV scanner. But a JsonL scanner, or TSV or some bson scanner (etc) are easy to consider as we collect more raw data.
