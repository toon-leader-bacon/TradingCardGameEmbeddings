# evaluation

Answers "how good is this embedding model, actually?" — structurally
similar to `training/` in that it can run any dojo's forward pass
end-to-end, but it never backpropagates or updates any weights. This
matters because dojo performance during training is optimization
pressure, not a trustworthy quality signal on its own; this container
is where quality actually gets measured, on held-out data, without
that pressure influencing the result.

Two kinds of work are expected to live here: running dojos' own
held-out splits through a trained encoder purely to score them (no
gradient step), and dojo-independent embedding-space exploration that
isn't tied to any one task at all — e.g. t-SNE or clustering plots
across games, nearest-neighbor sanity checks, or other ways of poking
at whether the embedding space looks sensible. The former reuses a
dojo's own contract read-only; the latter has no dependency on the
dojo interface at all, just a trained encoder and a corpus of cards.

No evaluation runner or exploration tooling exists yet — this file
grows as the first held-out scoring pass and the first embedding-space
visualization get built.
