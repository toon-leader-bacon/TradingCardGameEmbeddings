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
    PRINTED_NUMBER = "printed_number"  # a card number as printed on the card,
    # which deck lists name cards by in games where many cards share a name
    # (One Piece "OP01-024", Union Arena "UE01BT/BLC-1-041"); the id is the
    # number itself (see src/data_refinement/card_binder/tcgjson/).
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
    LIMITLESS_PLAY = "limitless_play"  # play.limitlesstcg.com's tournament
    # standings decklists (see src/data_retrieval/limitless_play/downloader.py
    # and src/data_refinement/deck_box/limitless_play/extraction_stage.py)
    # — distinct from POKEMON_TCG (that deck's cards' own data source).
    HEARTHSTONETOPDECKS = "hearthstonetopdecks"  # hearthstonetopdecks.com's
    # deck pages; source_id is the page's URL slug (see
    # src/data_retrieval/hearthstonetopdecks/downloader.py and
    # src/data_refinement/deck_box/hearthstonetopdecks/extraction_stage.py)
    # — distinct from HEARTHSTONEJSON (that deck's cards' own data source).
    ZENODO_HEARTHSTONE_DECKS = "zenodo_hearthstone_decks"  # the Zenodo
    # "Hearthstone Decks Dataset" (record 10198504); source_id is idDeck (see
    # src/data_retrieval/zenodo_hearthstone_decks/downloader.py and
    # src/data_refinement/deck_box/zenodo_hearthstone_decks/extraction_stage.py).
    MTGJSON_DECKS = "mtgjson_decks"  # MTGJSON's AllDeckFiles (Wizards
    # precons); source_id is the deck file's stem, e.g. "BoggartFeast_LRW"
    # (see src/data_retrieval/mtgjson_decks/downloader.py and
    # src/data_refinement/deck_box/mtgjson_decks/extraction_stage.py)
    # — distinct from SCRYFALL (that deck's cards' own data source).
    FABTCGMETA_ONLINE_DECKS = "fabtcgmeta_online_decks"  # fabtcgmeta.com's
    # online (Talishar) decklists; source_id is the deck's hash (see
    # src/data_retrieval/fabtcgmeta_online_decks/downloader.py and
    # src/data_refinement/deck_box/fabtcgmeta_online_decks/extraction_stage.py)
    # — distinct from CARDVAULT_FABTCG (that deck's cards' own data source).
    TCGJSON = "tcgjson"  # tcgjson's weekly TCGplayer catalog release;
    # source_id is a TCGplayer productId (one printing) (see
    # src/data_retrieval/tcgjson/downloader.py and
    # src/data_refinement/card_binder/tcgjson/ingestion_stage.py).
    SYSTEM = "system"  # minted internally by CardBinder itself for
    # synthetic sentinel cards (e.g. the "Unknown" fallback card — see
    # CardBinder.ensure_unknown_card()), never fetched from an external
    # source.
