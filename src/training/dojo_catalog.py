"""Every dojo a training run can name, and how to build it from disk.

A run config lists dojos by catalog key ("gwent_one.color_mask"). The key
becomes the dojo's `name`, so dojos from different games whose metric
files share a stem (gwent_one and dominiontabs both have set_mask) cannot
collide in split files, checkpoints or logs. Each entry is a recipe: which
game's CardBinder it needs, whether it needs a deck box, and which dojo
class to construct.

Loading is lazy and shared: a CardShelf loads each game's binder (and each
deck box) once, only when a named dojo needs it.

Contrastive dojos read a game's final deck box directly (no metric) and
take augmentation mods: the game's defaults
(src/dojos/augmentation_defaults.py) unless the run config overrides that
dojo's mods by name. Metric dojos build their own task mods and take no
augmentations yet.
"""

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Protocol, Sequence

from src.data_refinement.card_binder.card_binder import CardBinder
from src.data_refinement.deck_box.deck_box import DeckBox
from src.data_refinement.metrics.isotropic.deck_box_path import ISOTROPIC_DECK_BOX_PATH
from src.data_refinement.metrics.sts_gg.deck_box_path import STS_GG_DECK_BOX_PATH
from src.dojos.augmentation_defaults import default_augmentations_for
from src.dojos.contrastive.dojo import ContrastiveDojo
from src.dojos.contrastive.pair_constructor import SingleCardPairConstructor
from src.dojos.dojo import Dojo
from src.dojos.file_managers.deck_box_dealer import DeckBoxDealer
from src.dojos.final_decks import held_out_card_dojos as final_decks
from src.dojos.dominiontabs.cost_regression_dojo import CostRegressionDojo
from src.dojos.dominiontabs.masked_field_dojos import SetMaskDojo, TypeMaskDojo
from src.dojos.gwent_one import masked_field_dojos as gwent_one
from src.dojos import isotropic
from src.dojos.sts_gg import card_average_dojos as sts_cards
from src.dojos.mods.mod_pipeline import ModPipeline
from src.dojos.mods.mod_specs import ModSpec
from src.dojos.play_gwent.deck_card_mask_dojos import LeaderMaskedFromDeckDojo
from src.dojos.sts_gg import deck_label_dojos as sts_decks
from src.schema.game_id import GameId
from src.schema.holdout import HoldoutSpec

# Contrastive dojos keep their deck split index here (see DeckBoxDealer)
_CONTRASTIVE_INDEX_DIRECTORY = Path("data/splits/contrastive")


class CardDojoConstructor(Protocol):
    """A dojo class built from a CardBinder alone (every gwent_one and
    dominiontabs dojo, every sts_gg and isotropic card-level dojo)."""

    def __call__(
        self,
        card_binder: CardBinder,
        holdout: HoldoutSpec,
        card_embedding_size: int,
        *,
        name: str | None = None,
        rng_seed: int | None = None,
    ) -> Dojo: ...


class DeckDojoConstructor(Protocol):
    """A dojo class that also needs the DeckBox its metric was built from
    (every sts_gg and isotropic deck-level dojo, play_gwent's leader
    mask)."""

    def __call__(
        self,
        card_binder: CardBinder,
        holdout: HoldoutSpec,
        deck_box: DeckBox,
        card_embedding_size: int,
        *,
        name: str | None = None,
        rng_seed: int | None = None,
    ) -> Dojo: ...


class CardShelf:
    """Loads each game's CardBinder and each DeckBox at most once."""

    def __init__(self) -> None:
        """Side effects: none (nothing loads until asked)."""
        self._card_binders: dict[GameId, CardBinder] = {}
        self._deck_boxes: dict[Path, DeckBox] = {}

    def card_binder(self, game: GameId) -> CardBinder:
        """game's binder from CardBinder.default_output_path, loaded on
        first request.

        Inputs: game (GameId). Output: CardBinder.
        Side effects: reads the binder file on first call per game.
        Exceptions: FileNotFoundError if the binder file does not exist.

        Example:
            >>> CardShelf().card_binder(GameId.GWENT)
        """
        if game not in self._card_binders:
            path = CardBinder.default_output_path(game)
            if not path.exists():
                raise FileNotFoundError(
                    f"no {game.value} card binder at {path}; run "
                    "scripts/run_card_binder_ingestion.py first"
                )
            self._card_binders[game] = CardBinder.load([path])
        return self._card_binders[game]

    def deck_box(self, path: Path) -> DeckBox:
        """The DeckBox at path, loaded on first request.

        Inputs: path (Path). Output: DeckBox.
        Side effects: reads the deck box on first call per path.
        Exceptions: FileNotFoundError if path does not exist (DeckBox.load
            would silently return an empty box).

        Example:
            >>> CardShelf().deck_box(STS_GG_DECK_BOX_PATH)
        """
        if path not in self._deck_boxes:
            if not path.exists():
                raise FileNotFoundError(
                    f"no deck box at {path}; run scripts/run_metrics.py for "
                    "the metric source that writes it"
                )
            self._deck_boxes[path] = DeckBox.load([path])
        return self._deck_boxes[path]


@dataclass(frozen=True)
class DojoBuildContext:
    """What every recipe needs besides its own entry: the shared shelf and
    the run-wide settings every dojo must agree on.

    shelf: loaded card data. holdout: the plan's HoldoutSpec (the trainer
    requires every dojo to share it). card_embedding_size: the model's
    embed_dim. rng_seed: seeds each dojo's split shuffling, so a first
    split build is reproducible. mod_overrides: per dojo name,
    augmentation specs replacing that dojo's game defaults (an empty tuple
    means no augmentation); build_dojos rejects a name that is not built
    or takes no augmentations.
    """

    shelf: CardShelf
    holdout: HoldoutSpec
    card_embedding_size: int
    rng_seed: int
    mod_overrides: Mapping[str, tuple[ModSpec, ...]] = field(default_factory=dict)


class DojoRecipe(Protocol):
    """How to build one catalog dojo."""

    def build(self, name: str, context: DojoBuildContext) -> Dojo:
        """Construct the dojo, named name. Side effects: may load card
        data through context.shelf; may write split files on first build.
        Exceptions: FileNotFoundError for missing data; ValueError for a
        stale metric (the dojo's own version check)."""
        ...


@dataclass(frozen=True)
class CardDojoRecipe:
    """A dojo that needs only its game's CardBinder."""

    game: GameId
    dojo_class: CardDojoConstructor

    def build(self, name: str, context: DojoBuildContext) -> Dojo:
        """See DojoRecipe.build."""
        return self.dojo_class(
            context.shelf.card_binder(self.game),
            context.holdout,
            context.card_embedding_size,
            name=name,
            rng_seed=context.rng_seed,
        )


@dataclass(frozen=True)
class DeckDojoRecipe:
    """A dojo that needs its game's CardBinder and a DeckBox."""

    game: GameId
    dojo_class: DeckDojoConstructor
    deck_box_path: Path

    def build(self, name: str, context: DojoBuildContext) -> Dojo:
        """See DojoRecipe.build."""
        return self.dojo_class(
            context.shelf.card_binder(self.game),
            context.holdout,
            context.shelf.deck_box(self.deck_box_path),
            context.card_embedding_size,
            name=name,
            rng_seed=context.rng_seed,
        )


@dataclass(frozen=True)
class ContrastiveDojoRecipe:
    """A single-card contrastive dojo over one game's final deck box:
    positives are cards drawn from the same deck.

    game: whose binder and deck box. deck_box_path: the game's
    data/final/decks/<game>.db. items_per_deck / decks_per_sample: see
    SingleCardPairConstructor / ContrastiveDojo.
    """

    game: GameId
    deck_box_path: Path
    items_per_deck: int = 2
    decks_per_sample: int = 16

    def build(self, name: str, context: DojoBuildContext) -> Dojo:
        """See DojoRecipe.build. The dealer keeps its split index at
        data/splits/contrastive/<name>.db (seeded by context.rng_seed);
        mods come from context.mod_overrides[name] if present, else the
        game's defaults, each built with its own seed."""
        binder = context.shelf.card_binder(self.game)
        dealer = DeckBoxDealer(
            context.shelf.deck_box(self.deck_box_path),
            self.game,
            _contrastive_index_path(name),
            seed=context.rng_seed,
        )
        return ContrastiveDojo(
            dealer,
            SingleCardPairConstructor(self.items_per_deck, rng_seed=context.rng_seed),
            binder,
            context.holdout,
            decks_per_sample=self.decks_per_sample,
            name=name,
            mod_pipeline=_augmentation_pipeline(name, self.game, context),
        )


def _contrastive_index_path(name: str) -> Path:
    """Where a contrastive dojo's deck split index lives.
    Inputs: name. Output: Path. Side effects: none. Exceptions: none."""
    return _CONTRASTIVE_INDEX_DIRECTORY / f"{name}.db"


def _augmentation_pipeline(
    name: str, game: GameId, context: DojoBuildContext
) -> ModPipeline:
    """The dojo's mods: context.mod_overrides[name] if present, else
    game's defaults. Each mod's seed is drawn from a stream seeded by
    (rng_seed, name, "mods"), so mods neither share a random stream with
    each other nor with the dealer and pair constructor (seeded by
    rng_seed itself), and each dojo's mods differ.

    Inputs: name, game, context. Output: ModPipeline.
    Side effects: none. Exceptions: as a ModSpec's build.
    """
    specs = context.mod_overrides.get(name, default_augmentations_for(game))
    seeds = random.Random(f"{context.rng_seed}:{name}:mods")
    return ModPipeline([spec.build(seeds.randrange(2**32)) for spec in specs])


def _recipe_for_contrastive(game: GameId) -> ContrastiveDojoRecipe:
    """The contrastive recipe over game's final deck box, at
    DeckBox.default_output_path(game).

    Inputs: game. Output: ContrastiveDojoRecipe. Side effects: none.
    Exceptions: none.
    """
    return ContrastiveDojoRecipe(game, DeckBox.default_output_path(game))


def _recipe_for_gwent_one(dojo_class: CardDojoConstructor) -> CardDojoRecipe:
    """The recipe for a gwent_one card dojo class.

    Inputs: dojo_class. Output: its recipe. Side effects: none.
    Exceptions: none.
    """
    return CardDojoRecipe(GameId.GWENT, dojo_class)


def _recipe_for_sts_card(dojo_class: CardDojoConstructor) -> CardDojoRecipe:
    """The recipe for a sts_gg card-average dojo class.

    Inputs: dojo_class. Output: its recipe. Side effects: none.
    Exceptions: none.
    """
    return CardDojoRecipe(GameId.SLAY_THE_SPIRE_2, dojo_class)


def _recipe_for_sts_deck(dojo_class: DeckDojoConstructor) -> DeckDojoRecipe:
    """The recipe for a sts_gg deck-label dojo class.

    Inputs: dojo_class. Output: its recipe. Side effects: none.
    Exceptions: none.
    """
    return DeckDojoRecipe(GameId.SLAY_THE_SPIRE_2, dojo_class, STS_GG_DECK_BOX_PATH)


def _recipe_for_final_deck_box(
    game: GameId, dojo_class: DeckDojoConstructor
) -> DeckDojoRecipe:
    """The recipe for a dojo whose metric rows point into game's
    published deck box, at DeckBox.default_output_path(game).

    Inputs: game, dojo_class. Output: DeckDojoRecipe. Side effects: none.
    Exceptions: none.
    """
    return DeckDojoRecipe(game, dojo_class, DeckBox.default_output_path(game))


def _recipe_for_isotropic_deck(dojo_class: DeckDojoConstructor) -> DeckDojoRecipe:
    """The recipe for an isotropic deck-level dojo class.
    Inputs: dojo_class. Output: DeckDojoRecipe. Side effects: none.
    Exceptions: none."""
    return DeckDojoRecipe(GameId.DOMINION, dojo_class, ISOTROPIC_DECK_BOX_PATH)


# Keys are "<metric source>.<metric file stem>"; add a line to onboard a dojo
DOJO_CATALOG: Mapping[str, DojoRecipe] = {
    "gwent_one.armor_mask": _recipe_for_gwent_one(gwent_one.ArmorMaskDojo),
    "gwent_one.color_mask": _recipe_for_gwent_one(gwent_one.ColorMaskDojo),
    "gwent_one.faction_mask": _recipe_for_gwent_one(gwent_one.FactionMaskDojo),
    "gwent_one.power_mask": _recipe_for_gwent_one(gwent_one.PowerMaskDojo),
    "gwent_one.provision_mask": _recipe_for_gwent_one(gwent_one.ProvisionMaskDojo),
    "gwent_one.rarity_mask": _recipe_for_gwent_one(gwent_one.RarityMaskDojo),
    "gwent_one.set_mask": _recipe_for_gwent_one(gwent_one.SetMaskDojo),
    "gwent_one.type_mask": _recipe_for_gwent_one(gwent_one.TypeMaskDojo),
    "dominiontabs.cost_regression": CardDojoRecipe(GameId.DOMINION, CostRegressionDojo),
    "dominiontabs.set_mask": CardDojoRecipe(GameId.DOMINION, SetMaskDojo),
    "dominiontabs.type_mask": CardDojoRecipe(GameId.DOMINION, TypeMaskDojo),
    "isotropic.average_copies_bought": CardDojoRecipe(
        GameId.DOMINION, isotropic.AverageCopiesBoughtDojo
    ),
    "isotropic.opening_buy_rate": CardDojoRecipe(
        GameId.DOMINION, isotropic.OpeningBuyRateDojo
    ),
    "isotropic.pile_exhaustion_rate": CardDojoRecipe(
        GameId.DOMINION, isotropic.PileExhaustionRateDojo
    ),
    "isotropic.turn_count_association": CardDojoRecipe(
        GameId.DOMINION, isotropic.TurnCountAssociationDojo
    ),
    "isotropic.full_deck_win_prediction": _recipe_for_isotropic_deck(
        isotropic.FullDeckWinPredictionDojo
    ),
    "isotropic.kingdom_ending_type": _recipe_for_isotropic_deck(
        isotropic.KingdomEndingTypeDojo
    ),
    "isotropic.kingdom_game_length": _recipe_for_isotropic_deck(
        isotropic.KingdomGameLengthDojo
    ),
    "isotropic.mid_game_next_turn_action_count": _recipe_for_isotropic_deck(
        isotropic.NextTurnActionCountDojo
    ),
    "isotropic.winning_deck_masked_card": _recipe_for_isotropic_deck(
        isotropic.WinningDeckMaskedCardDojo
    ),
    "isotropic.veto_rate": CardDojoRecipe(GameId.DOMINION, isotropic.VetoRateDojo),
    "isotropic.kingdom_opening_buy_prediction": _recipe_for_isotropic_deck(
        isotropic.KingdomOpeningBuyDojo
    ),
    "isotropic.kingdom_veto_prediction": _recipe_for_isotropic_deck(
        isotropic.KingdomVetoDojo
    ),
    "isotropic.mid_game_next_buy": _recipe_for_isotropic_deck(isotropic.NextBuyDojo),
    "isotropic.mid_game_next_trashed_card": _recipe_for_isotropic_deck(
        isotropic.NextTrashedCardDojo
    ),
    "isotropic.deck_pair_winner": _recipe_for_isotropic_deck(
        isotropic.DeckPairWinnerDojo
    ),
    "isotropic.mid_game_deck_pair_winner": _recipe_for_isotropic_deck(
        isotropic.MidGameDeckPairWinnerDojo
    ),
    "isotropic.mid_game_win_probability": _recipe_for_isotropic_deck(
        isotropic.EventualWinDojo
    ),
    "isotropic.opening_buy_outcome": _recipe_for_isotropic_deck(
        isotropic.OpeningBuyOutcomeDojo
    ),
    "isotropic.winning_deck_membership": _recipe_for_isotropic_deck(
        isotropic.WinningDeckMembershipDojo
    ),
    "isotropic.kingdom_ending_pile_prediction": _recipe_for_isotropic_deck(
        isotropic.KingdomEndingPileDojo
    ),
    "isotropic.winning_deck_count": _recipe_for_isotropic_deck(
        isotropic.WinningDeckCountDojo
    ),
    "isotropic.deck_card_set_copy_count": _recipe_for_isotropic_deck(
        isotropic.DeckCardSetCopyCountDojo
    ),
    # play_gwent writes its decks into the final gwent deck box
    "play_gwent.leader_masked_from_deck": DeckDojoRecipe(
        GameId.GWENT,
        LeaderMaskedFromDeckDojo,
        DeckBox.default_output_path(GameId.GWENT),
    ),
    "sts_gg.card_deck_size": _recipe_for_sts_card(sts_cards.CardDeckSizeDojo),
    "sts_gg.card_elites_killed": _recipe_for_sts_card(sts_cards.CardElitesKilledDojo),
    "sts_gg.card_floors_cleared": _recipe_for_sts_card(sts_cards.CardFloorsClearedDojo),
    "sts_gg.card_relic_count": _recipe_for_sts_card(sts_cards.CardRelicCountDojo),
    "sts_gg.card_total_cards_picked": _recipe_for_sts_card(
        sts_cards.CardTotalCardsPickedDojo
    ),
    "sts_gg.card_total_combats": _recipe_for_sts_card(sts_cards.CardTotalCombatsDojo),
    "sts_gg.card_total_damage_taken": _recipe_for_sts_card(
        sts_cards.CardTotalDamageTakenDojo
    ),
    "sts_gg.card_total_turns": _recipe_for_sts_card(sts_cards.CardTotalTurnsDojo),
    "sts_gg.card_win_rate": _recipe_for_sts_card(sts_cards.CardWinRateDojo),
    "sts_gg.character_prediction": _recipe_for_sts_deck(sts_decks.CharacterDojo),
    "sts_gg.elites_killed": _recipe_for_sts_deck(sts_decks.DeckElitesKilledDojo),
    "sts_gg.floors_cleared": _recipe_for_sts_deck(sts_decks.DeckFloorsClearedDojo),
    "sts_gg.relic_count": _recipe_for_sts_deck(sts_decks.DeckRelicCountDojo),
    "sts_gg.total_cards_picked": _recipe_for_sts_deck(
        sts_decks.DeckTotalCardsPickedDojo
    ),
    "sts_gg.total_cards_skipped": _recipe_for_sts_deck(
        sts_decks.DeckTotalCardsSkippedDojo
    ),
    "sts_gg.total_combats": _recipe_for_sts_deck(sts_decks.DeckTotalCombatsDojo),
    "sts_gg.total_damage_taken": _recipe_for_sts_deck(
        sts_decks.DeckTotalDamageTakenDojo
    ),
    "sts_gg.total_turns": _recipe_for_sts_deck(sts_decks.DeckTotalTurnsDojo),
    "sts_gg.win": _recipe_for_sts_deck(sts_decks.WinDojo),
    "final_decks.held_out_card_pokemon": _recipe_for_final_deck_box(
        GameId.POKEMON, final_decks.PokemonHeldOutCardDojo
    ),
    "final_decks.held_out_card_flesh_and_blood": _recipe_for_final_deck_box(
        GameId.FLESH_AND_BLOOD, final_decks.FleshAndBloodHeldOutCardDojo
    ),
    "final_decks.held_out_card_gwent": _recipe_for_final_deck_box(
        GameId.GWENT, final_decks.GwentHeldOutCardDojo
    ),
    "final_decks.held_out_card_dominion": _recipe_for_final_deck_box(
        GameId.DOMINION, final_decks.DominionHeldOutCardDojo
    ),
    "final_decks.held_out_card_slay_the_spire_2": _recipe_for_final_deck_box(
        GameId.SLAY_THE_SPIRE_2, final_decks.SlayTheSpire2HeldOutCardDojo
    ),
    "final_decks.held_out_card_mtg": _recipe_for_final_deck_box(
        GameId.MTG, final_decks.MtgHeldOutCardDojo
    ),
    "contrastive.gwent": _recipe_for_contrastive(GameId.GWENT),
    "contrastive.flesh_and_blood": _recipe_for_contrastive(GameId.FLESH_AND_BLOOD),
    "contrastive.slay_the_spire_2": _recipe_for_contrastive(GameId.SLAY_THE_SPIRE_2),
    "contrastive.mtg": _recipe_for_contrastive(GameId.MTG),
    "contrastive.pokemon": _recipe_for_contrastive(GameId.POKEMON),
    "contrastive.dominion": _recipe_for_contrastive(GameId.DOMINION),
}


def takes_augmentations(name: str) -> bool:
    """Whether the catalog dojo named name accepts augmentation mods
    (today: the contrastive dojos). build_dojos uses this to reject a
    `mods:` override for any other dojo.

    Inputs: name (a DOJO_CATALOG key). Output: bool.
    Side effects: none. Exceptions: KeyError for an unknown name.

    Example:
        >>> takes_augmentations("contrastive.gwent")
        True
    """
    return isinstance(DOJO_CATALOG[name], ContrastiveDojoRecipe)


def build_dojos(names: Sequence[str], context: DojoBuildContext) -> list[Dojo]:
    """Build every named catalog dojo, in order.

    Inputs: names (Sequence[str], DOJO_CATALOG keys), context
        (DojoBuildContext).
    Output: list[Dojo], same order as names, each named by its key.
    Side effects: loads card data through context.shelf; a dojo whose
        split files do not exist yet writes them (seeded by
        context.rng_seed).
    Exceptions: ValueError naming every unknown key, or any
        context.mod_overrides name that is not in names or takes no
        augmentations, raised before any dojo is built (so a typo does
        not cost a binder load); whatever a recipe's build raises.

    Example:
        >>> context = DojoBuildContext(CardShelf(), holdout, 256, rng_seed=0)
        >>> build_dojos(["gwent_one.color_mask"], context)[0].name
        'gwent_one.color_mask'
    """
    result: list[Dojo] = []

    # Validate inputs: every name is in the catalog
    unknown = [name for name in names if name not in DOJO_CATALOG]
    if unknown:
        raise ValueError(f"unknown dojos {unknown}; see DOJO_CATALOG")
    _require_valid_overrides(context.mod_overrides, names)

    # Build each dojo from its recipe
    for name in names:
        result.append(DOJO_CATALOG[name].build(name, context))
    return result


def _require_valid_overrides(
    mod_overrides: Mapping[str, tuple[ModSpec, ...]], names: Sequence[str]
) -> None:
    """Inputs: mod_overrides, names (the dojos being built). Output: none.
    Side effects: none.
    Exceptions: ValueError if an override names a dojo not being built, or
        one that takes no augmentations (its override would be ignored).
    """
    not_built = sorted(set(mod_overrides) - set(names))
    if not_built:
        raise ValueError(f"mods override dojos {not_built} that are not being built")
    fixed = sorted(name for name in mod_overrides if not takes_augmentations(name))
    if fixed:
        raise ValueError(f"dojos {fixed} take no augmentation mods")
