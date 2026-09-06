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
