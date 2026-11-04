"""ContrastiveDojoRecipe: a catalog contrastive dojo over one game's final
deck box (read through its level-2 deck snapshot), with its split index
and optional staple subsampling."""

from dataclasses import dataclass
from pathlib import Path

from src.dojos.contrastive.contrastive_style import ContrastiveStyle
from src.dojos.contrastive.dojo import ContrastiveDojo
from src.dojos.contrastive.staple_subsampling import (
    StapleSubsampling,
    cached_document_frequency,
)
from src.dojos.contrastive.styles.single_card import SingleCardStyle
from src.dojos.dojo import Dojo
from src.dojos.file_managers.deck_box_dealer import DeckBoxDealer
from src.dojos.file_managers.deck_split_index import deck_split_tag
from src.schema.game_id import GameId
from src.training.dojo_recipes.augmentations import build_augmentation_pipeline
from src.training.dojo_recipes.dojo_build_context import DojoBuildContext

# TRAIN decks a staple-subsampling document frequency is counted over
_DOCUMENT_FREQUENCY_SAMPLE_DECKS = 20_000


@dataclass(frozen=True)
class ContrastiveDojoRecipe:
    """A contrastive dojo over one game's final deck box, in one
    contrastive style (contrastive_style.py).

    game: whose binder and deck box. deck_box_path: the game's
    data/final/decks/<game>.db; the dojo deals from the game's deck
    snapshot, checked against this box. style: builds the matched pair
    constructor and loss (default: single cards, every same-deck card a
    positive). The run's batch budget alone sets decks per batch (see
    ContrastiveDojo.batches).
    """

    game: GameId
    deck_box_path: Path
    style: ContrastiveStyle = SingleCardStyle()

    def build(self, name: str, context: DojoBuildContext) -> Dojo:
        """See DojoRecipe.build. The dealer deals from the game's deck
        snapshot (checked against deck_box_path's box, so a missing or
        stale snapshot fails here, before preflight), split by the
        shelf's split index for context.holdout, cached in the shelf's
        deck split cache directory and shared by every style of the game;
        TRAIN reshuffles from context.rng_seed. Mods come from
        context.mod_overrides[name] if present, else the game's
        defaults, each built with its own seed. A
        context.staple_thresholds[name] turns on staple subsampling, its
        document frequency cached beside the split index."""
        binder = context.shelf.card_binder(self.game)
        # Both levels are checked here: the box exists, the snapshot matches it
        snapshot = context.shelf.deck_snapshot(self.game, self.deck_box_path)
        split_index = context.shelf.deck_split_index(snapshot, context.holdout)
        dealer = DeckBoxDealer(snapshot, split_index, context.rng_seed)
        # The style builds the pair constructor and loss as a matched set
        pair_constructor = self.style.pair_constructor(
            context.rng_seed, _staple_subsampling(name, dealer, context)
        )
        return ContrastiveDojo(
            dealer,
            pair_constructor,
            binder,
            context.holdout,
            contrastive_loss=self.style.contrastive_loss(),
            name=name,
            mod_pipeline=build_augmentation_pipeline(name, self.game, context),
        )


def _staple_subsampling(
    name: str, dealer: DeckBoxDealer, context: DojoBuildContext
) -> StapleSubsampling | None:
    """name's staple subsampling, or None (t = inf) when the run config
    names no threshold for it. Its document frequency comes from
    cached_document_frequency, cached at
    <shelf's deck split cache directory>/<name>.<split tag>.document_frequency.json.

    Inputs: name, dealer (the dojo's), context.
    Output: StapleSubsampling | None.
    Side effects: may count and cache the document frequency (reads decks
        through dealer; writes only the cache file).
    Exceptions: as cached_document_frequency and StapleSubsampling.
    """
    threshold = context.staple_thresholds.get(name)
    if threshold is None:
        return None
    split_tag = deck_split_tag(dealer.deck_split_key)
    cache_path = (
        context.shelf.deck_split_cache_directory
        / f"{name}.{split_tag}.document_frequency.json"
    )
    frequency = cached_document_frequency(
        dealer, cache_path, _DOCUMENT_FREQUENCY_SAMPLE_DECKS
    )
    return StapleSubsampling(threshold, frequency)
