"""ContrastiveDojoRecipe: a catalog contrastive dojo over one game's final
deck box, with its split index and optional staple subsampling."""

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
from src.schema.game_id import GameId
from src.training.dojo_recipes.augmentations import build_augmentation_pipeline
from src.training.dojo_recipes.dojo_build_context import DojoBuildContext

# Contrastive dojos keep their deck split index here (see DeckBoxDealer)
_CONTRASTIVE_INDEX_DIRECTORY = Path("data/splits/contrastive")

# TRAIN decks a staple-subsampling document frequency is counted over
_DOCUMENT_FREQUENCY_SAMPLE_DECKS = 20_000


@dataclass(frozen=True)
class ContrastiveDojoRecipe:
    """A contrastive dojo over one game's final deck box, in one
    contrastive style (contrastive_style.py).

    game: whose binder and deck box. deck_box_path: the game's
    data/final/decks/<game>.db. style: builds the matched pair
    constructor and loss (default: single cards, every same-deck card a
    positive). decks_per_sample: see ContrastiveDojo.
    """

    game: GameId
    deck_box_path: Path
    style: ContrastiveStyle = SingleCardStyle()
    decks_per_sample: int = 16

    def build(self, name: str, context: DojoBuildContext) -> Dojo:
        """See DojoRecipe.build. The dealer keeps its split index at
        data/splits/contrastive/<name>.db (seeded by context.rng_seed);
        mods come from context.mod_overrides[name] if present, else the
        game's defaults, each built with its own seed. A
        context.staple_thresholds[name] turns on staple subsampling, its
        document frequency cached beside the split index."""
        binder = context.shelf.card_binder(self.game)
        dealer = DeckBoxDealer(
            context.shelf.deck_box(self.deck_box_path),
            self.game,
            _contrastive_index_path(name),
            seed=context.rng_seed,
        )
        # The style builds the pair constructor and loss as a matched set
        pair_constructor = self.style.pair_constructor(
            context.rng_seed, _staple_subsampling(name, dealer, context)
        )
        return ContrastiveDojo(
            dealer,
            pair_constructor,
            binder,
            context.holdout,
            decks_per_sample=self.decks_per_sample,
            contrastive_loss=self.style.contrastive_loss(),
            name=name,
            mod_pipeline=build_augmentation_pipeline(name, self.game, context),
        )


def _contrastive_index_path(name: str) -> Path:
    """Where a contrastive dojo's deck split index lives.
    Inputs: name. Output: Path. Side effects: none. Exceptions: none."""
    return _CONTRASTIVE_INDEX_DIRECTORY / f"{name}.db"


def _staple_subsampling(
    name: str, dealer: DeckBoxDealer, context: DojoBuildContext
) -> StapleSubsampling | None:
    """name's staple subsampling, or None (t = inf) when the run config
    names no threshold for it. Its document frequency comes from
    cached_document_frequency, cached at
    data/splits/contrastive/<name>.document_frequency.json.

    Inputs: name, dealer (the dojo's), context.
    Output: StapleSubsampling | None.
    Side effects: may count and cache the document frequency (reads decks
        through dealer; writes only the cache file).
    Exceptions: as cached_document_frequency and StapleSubsampling.
    """
    threshold = context.staple_thresholds.get(name)
    if threshold is None:
        return None
    cache_path = _CONTRASTIVE_INDEX_DIRECTORY / f"{name}.document_frequency.json"
    frequency = cached_document_frequency(
        dealer, cache_path, _DOCUMENT_FREQUENCY_SAMPLE_DECKS
    )
    return StapleSubsampling(threshold, frequency)
