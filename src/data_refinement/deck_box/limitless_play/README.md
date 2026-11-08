# `limitless_play/`: Limitless tournament decklists

These stages turn the standings that `src/data_retrieval/limitless_play/`
downloads (`data/raw/limitless_play/<game>/standings.jsonl`) into deck boxes
(`data/final/decks/<game>.db`), one stage per card game:

- `LimitlessPlayDeckExtractionStage` (Pokemon TCG, `--source limitless_play`)
  is in `extraction_stage.py`.
- The other eight games' stages are in `tcg_extraction_stages.py`
  (`--source limitless_<game>`, e.g. `limitless_one_piece`).

## Files

- `limitless_decklist.py` parses a standings row into one or more
  `LimitlessDecklist`s (one per deck in a multi-deck tournament). A
  single-card section (`leader`, `base`) is one copy. The `side` section is
  the sideboard and is dropped, as `ygoprodeck_decks` and `mtgjson_decks`
  drop sideboards.
- `extraction_stage.py` holds `LimitlessDeckExtractionStage`, the loop every
  game shares:
  - it skips lists whose card count is outside the game's `DeckSizes`
    (`../deck_sizes.py`);
  - it matches each entry through the game's card matcher (a
    `CachedCardMatcher` from `../card_matching.py`, keyed by `LimitlessCardRef`),
    using the Unknown sentinel when nothing matches;
  - it stores multiset decks, so identical lists are stored once;
  - it logs decklist outcomes and match counts.

  The same file holds the Pokemon stage.
- `card_index.py` matches Pokemon cards. It maps Limitless's set code to
  pokemon-tcg-data set ids (via pokemon-tcg-data's `sets/en.json`), looks up
  `<set id>-<number>` as a `POKEMON_TCG` alias, then falls back to the newest
  card with the same name.
- `tcg_card_index.py` matches every other game's cards. It first looks up
  `<set>-<number>` (e.g. `OP13-001`) as a `PRINTED_NUMBER` alias (number games
  only), then the folded name. Folding drops diacritics, zero-width spaces and
  quote styles, and reads `", "` as `" - "`, so "Sabine Wren, Explosives
  Artist" finds "Sabine Wren - Explosives Artist". A folded name that several
  binder cards share matches nothing, unlike Pokemon's newest-card rule:
  Digimon alone prints many different "Agumon".
- `game_spec.py` holds one `LimitlessDeckSpec` per non-Pokemon game. A spec
  gives the deck sizes and whether the game matches by number.
  `tcg_extraction_stages.py` builds one stage class per spec.

## Status per Limitless game

Deck sizes count every section except `side`.

| Limitless game (`--game`) | `GameId` | Deck sizes | Card match | Deck stage (`--source`) |
|---|---|---|---|---|
| Pokemon TCG (`PTCG`) | `POKEMON` | 60 | Pokemon printing, then newest same name | `limitless_play` |
| One Piece (`OP`) | `ONE_PIECE` | 51 (leader + 50) | number, then name | `limitless_one_piece` |
| Digimon (`DCG`) | `DIGIMON` | 50-55 (50 + 0-5 Digi-Eggs) | number, then name | `limitless_digimon` |
| Gundam (`GUNDAM`) | `GUNDAM` | 50 | number, then name | `limitless_gundam` |
| Fusion World (`FW`) | `DRAGON_BALL_FUSION_WORLD` | 51-61 (leader + 50-60) | number, then name | `limitless_dragon_ball_fusion_world` |
| Dragon Ball Super Masters (`DBS`) | `DRAGON_BALL_MASTERS` | 51 or more (leader + main + Z-deck) | number, then name | `limitless_dragon_ball_masters` |
| Battle Spirits Saga (`BSS`) | `BATTLE_SPIRITS_SAGA` | 50-60 | number, then name | `limitless_battle_spirits_saga` |
| Star Wars: Unlimited (`SWU`) | `STAR_WARS_UNLIMITED` | 52 or more (leader + base + 50) | name | `limitless_star_wars_unlimited` |
| Lorcana (`LORCANA`) | `LORCANA` | 60 or more | name | `limitless_lorcana` |
| Riftbound (`RB`) | `RIFTBOUND` | – | – | none: its standings carry no decklists |
| Pokemon TCG Pocket (`POCKET`) | none | – | – | none: needs a card binder first |
| Pokemon VGC (`VGC`), Pokemon GO (`PGO`) | none | – | – | out of scope: teams, not card decks |

Union Arena (`UNION_ARENA`) has a binder, but Limitless doesn't host
tournaments for it.
