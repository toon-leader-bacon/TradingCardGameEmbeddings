"""The fixed-but-growing set of external systems a card identifier can come from.

Backs Provenance/AliasLedger's multi-source identity resolution (see
src/data_refinement/card_binder/README.md) — a single card can carry
several external identifiers across several DataSources at once (e.g.
Scryfall's oracle_id and Arena's numeric id), which is why identity
resolution lives in AliasLedger rather than a single field. Mirrors
GameId's shape (str, Enum) and extensibility story.
"""

from enum import Enum


class DataSource(str, Enum):
    """Which external system minted a given card identifier.

    Extensible per new source onboarded — e.g. a future MTGO-specific
    downloader, or a second game's marketplace, adds a case here.

    Inputs: none (enum).
    Output: n/a.
    Side effects: none.
    Exceptions: none.
    """

    SCRYFALL = "scryfall"
    ARENA = "arena"
    MTGO = "mtgo"
    PRINTED_NAME = "printed_name"  # a name a card is printed or spelled under
    # that differs from its canonical name (MTG: Arena's names for OM1 cards,
    # from Scryfall's printed_name); the id is the name itself.
    GATHERER = "gatherer"  # Wizards' Gatherer database id (Scryfall's
    # multiverse_ids field) — added during CardBinder implementation,
    # not present in the original skeleton's enum members; same
    # extensibility story as the other members.
    POKEMON_TCG = "pokemon_tcg"  # pokemon-tcg-data community GitHub
    # repo (see src/data_retrieval/pokemon_tcg/downloader.py and
    # src/data_refinement/card_binder/pokemon_tcg/ingestion_stage.py).
    GWENT_ONE = "gwent_one"  # gwent.one's card search AJAX endpoint
    # (see src/data_retrieval/gwent_one/downloader.py and
    # src/data_refinement/card_binder/gwent_one/ingestion_stage.py).
    SPIRE_CODEX = "spire_codex"  # spire-codex GitHub repo's cards.json
    # (see src/data_retrieval/spire_codex/downloader.py and
    # src/data_refinement/card_binder/spire_codex/ingestion_stage.py).
    CARDVAULT_FABTCG = "cardvault_fabtcg"  # cardvault.fabtcg.com's
    # public_card_data.csv dump (see
    # src/data_retrieval/cardvault_fabtcg/card_downloader.py and
    # src/data_refinement/card_binder/cardvault_fabtcg/ingestion_stage.py).
    FABTCG_DECKLISTS = "fabtcg_decklists"  # fabtcg.com's own decklist
    # HTML fragments (see src/data_retrieval/fabtcg_decklists/downloader.py
    # and src/data_refinement/deck_box/fabtcg_decklists/extraction_stage.py)
    # — the raw deck source itself, distinct from CARDVAULT_FABTCG (that
    # deck's cards' own data source).
    PLAY_GWENT = "play_gwent"  # playgwent.com's guides.jsonl deck
    # guides (see src/data_retrieval/play_gwent/downloader.py and
    # src/data_refinement/deck_box/play_gwent/extraction_stage.py) —
    # distinct from GWENT_ONE (that deck's cards' own data source).
    STS2RUNS = "sts2runs"  # sts2runs.com's monthly community run dump. The
    # source is dead and its code is in archive/sts2runs/; the value stays
    # because 6,796 decks in the StS2 deck box carry it as provenance.
    STS_GG = "sts_gg"  # sts_gg's runs.jsonl run data (see
    # src/data_retrieval/sts_gg/run_downloader.py and
    # src/data_refinement/deck_box/sts_gg/extraction_stage.py).
    SEVENTEENLANDS_GAME_DATA = "seventeenlands_game_data"  # 17Lands'
    # per-set/per-format game_data CSVs' constructed decks (see
    # src/data_retrieval/seventeenlands/downloader.py and
    # src/data_refinement/deck_box/seventeenlands_game_data/extraction_stage.py)
    # — distinct from each deck's own cards' data source (SCRYFALL, via
    # card_binder/scryfall/).
    HEARTHSTONEJSON = "hearthstonejson"  # HearthstoneJSON per-build
    # cards.json snapshots; source_id is a card's dbfId (see
    # src/data_retrieval/hearthstonejson/downloader.py and
    # src/data_refinement/card_binder/hearthstonejson/ingestion_stage.py).
    DOMINIONTABS = "dominiontabs"  # sumpfork/dominiontabs' card_db_src
    # JSON files (see src/data_retrieval/dominiontabs/downloader.py and
    # src/data_refinement/card_binder/dominiontabs/ingestion_stage.py).
    ISOTROPIC = "isotropic"  # isotropic.org's Wayback-salvaged Dominion
    # gamelog summary JSON (see
    # src/data_retrieval/isotropic/downloader.py and
    # src/data_refinement/metrics/isotropic/BRAINSTORM.md) — this
    # source's decks/kingdoms are minted by its own metrics directly
    # (no card_binder ingestion stage: card names resolve against the
    # already-ingested DOMINIONTABS binder instead).
    SYSTEM = "system"  # minted internally by CardBinder itself for
    # synthetic sentinel cards (e.g. the "Unknown" fallback card — see
    # CardBinder.ensure_unknown_card()), never fetched from an external
    # source.
