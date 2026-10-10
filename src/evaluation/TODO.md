# evaluation TODO

## Card in contexts: identity retention vs. context sensitivity

Use the card-in-contexts contrastive dojo
(`contrastive_card_in_contexts.<game>`,
`src/dojos/contrastive/styles/card_in_contexts.py`) as an evaluation
probe, and build metrics and visualizations from what it finds.

The dojo places one anchor card in two disjoint slices of its deck and asks
whether the anchor's two contextual embeddings still match each other
better than other decks' anchors. Read alone, a good score is ambiguous:
a model that ignores context scores perfectly, and `SingleCardModel` does by
construction. So report it as one axis of a pair:

- **Identity retention**: card-in-contexts loss (normalized by its
  baseline) and top-1 accuracy on TEST batches. Does a card stay
  recognizably itself across decks?
- **Context sensitivity**: how far a card's contextual embedding moves
  away from its isolated one (cosine distance, averaged over contexts), and
  how much its two contexts disagree. Does context change anything at all?

A healthy multi-card model scores well on both. Collapse (context ignored)
shows high retention and near-zero sensitivity. Over-contextualization
(the card overwritten by its deck) shows the reverse. `SingleCardModel`
is the reference point: perfect retention, zero sensitivity.

Ideas:

- A scalar pair per encoder and game, written like the other analyses'
  `scalars.json`.
- A retention-vs-sensitivity scatter: one point per encoder (or per
  checkpoint over a run, to watch it drift), per game.
- Per-card breakdown: which cards lose their identity most in context
  (likely flexible staples), and which never move (likely vanilla or
  narrow cards).
- A small-multiples view of one anchor's contextual embeddings across many
  decks, projected next to its isolated embedding.

Card in contexts stays out of every training diet; see the dojo's
docstring.
