# intrinsic

Properties of the embedding space itself, measured directly - no
downstream model is trained. Input to both workflows: a corpus of
embeddings plus a `card -> label` mapping (see `../README.md`'s "shared
building block"). The corpus can be one game, several, or every game;
the label source is a metric's own output (e.g. `ColorMaskMetric`), not
a `Dojo`.

Cheap enough to run against many checkpoints, or even mid-training,
since nothing here does gradient descent.

## Two workflows

**1. Cluster-then-match** (does the space *discover* this structure
unsupervised). Cluster the embeddings with no knowledge of the true
labels (KMeans with k = number of known classes, or HDBSCAN if k is
unknown), then score how well the discovered clusters agree with the
true labels using a permutation-invariant agreement metric - adjusted
Rand index or normalized mutual information - rather than requiring
cluster 3 to literally be named "Red". This is the stronger, more
skeptical claim: nothing told the space this structure should exist.

**2. Known-labels-then-tightness** (given the true labels, how *cleanly
separated* are they already). No clustering algorithm - group embeddings
by their already-known true label, then measure how much tighter each
group is around its own centroid than around every other group's, via
`silhouette_score(embeddings, labels)` (or the cheaper, more
interpretable nearest-centroid accuracy: assign each card to whichever
known label's centroid is nearest, and see how often that matches its
real label).

Both workflows should be reported alongside a t-SNE or UMAP 2D
projection of the same embeddings, colored by the same label - the
projection is for human eyes and is not itself a metric (t-SNE distorts
global distances and is well known to show visually convincing structure
even in noise), so the plot illustrates what the ARI/NMI/silhouette
number already established quantitatively, never substitutes for it.
Fix the projection's seed and perplexity/n_neighbors so plots are
reproducible across checkpoints.

## Status

Design only; no code yet.
