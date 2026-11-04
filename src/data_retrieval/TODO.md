# TODO (data retrieval)

## Pokemon

Player-made decklist sources, surveyed 2026-10-28 (the full write-up is
in `plans/pokemon_deck_sources.md`, local only). The first,
the Limitless play API, is built (`limitless_play/`), covering ~17k
tournaments. `plans/pre_training_data.md` (local only) still lists
Pokemon decks as "nothing more planned" and needs updating.

- [ ] **Limitless play API key.** Check on the API key application
  (https://play.limitlesstcg.com/account/settings/api). A key lifts the
  keyless 50-requests-per-5-min limit that makes a full re-crawl take
  ~30 h.
- [ ] **limitlesstcg.com tournament database.** ~25-30k decklists from
  ~600 major (mostly in-person) events. HTML pages (`/decks/list/<id>`)
  with `data-set`/`data-number` per card, the same set-code scheme as
  `limitless_play/`, so its card matcher can be reused. robots.txt
  allows everything. Some overlap with the play API's big online events.
  About 8 h at 60/min.
- [ ] **Japanese official deck codes (pokemon-card.com, via
  pokecabook.com).** Likely tens of thousands of City League lists.
  robots allow both. Two hops: harvest codes from result pages, then
  fetch each deck. Blocker: cards are Japanese printings, and we have no
  JP-to-EN card map. The hardest one here.
- [ ] **Dittobase and TCG Stacked.** Small community deck builders,
  hundreds of decks at best (Dittobase's deck sitemap has ~520 pages,
  TCG Stacked's ~20). Only worth it for non-tournament diversity, after
  the two above.

Considered and rejected:

- **pokemoncard.io user decks.** Probably the largest user deck
  builder, but its robots.txt says `ai-train=no` (reserved under EU DSM
  Art. 4). Not to be used unless that changes.
- **RK9 Labs** (official event software). robots.txt disallows
  `/decklist/public/` and `/pairings/`.
- **TCGplayer Infinite decks.** robots.txt disallows deck pages.
- **justinbasil.com** (historic Worlds lists). robots.txt blocks
  `anthropic-ai`, `ClaudeBot`, `GPTBot` and other AI crawlers.
- **TrainerHill.** Aggregates Limitless data; nothing new.
- **Apify "Limitless scraper".** A paid wrapper around the play API.
- **Pokemon TCG Pocket sites.** A different game with a different card
  pool.

## Hearthstone

Player-made decklist sources, surveyed 2026-10-28 (full write-up:
`plans/hearthstone_deck_sources.md`, local only). The first,
HearthstoneTopDecks (~119k deck pages), is built
(`hearthstonetopdecks/`). Sources that publish deck codes match the
binder exactly by dbfId; name-only sources need name matching.

The Zenodo "Hearthstone Decks Dataset" (DOI 10.5281/zenodo.10198504,
852k decks, 2013-2020, CC BY 4.0) is also built
(`zenodo_hearthstone_decks/`). Its decks came from HearthPwn, which now
says `ai-train=no` (below). On 2026-10-28 the user chose to use it
anyway: it is the data of Mora et al. (2022), so results can be compared
with that paper.

- [ ] **Zenodo set: decide training vs. evaluation use.** Option: evaluate
  on it (its archetype labels) and keep it out of training, given the
  HearthPwn provenance. Deck provenance keeps only the first source to
  store a list, so a Zenodo-only split must be rebuilt from `cards.csv`.
- [ ] **outof.games.** ~44k Hearthstone decks (44 sitemap pages),
  robots allow. Not yet checked for a deck code on the page.
- [ ] **hearthstone-decks.net.** ~40k legend-ladder deck posts with deck
  codes, robots allow. Likely overlaps HearthstoneTopDecks.
- [ ] **Vicious Syndicate deck library.** ~5k curated, archetype-labelled
  lists (3 deck sitemaps), robots allow.

Considered and rejected:

- **HearthPwn.** The largest user deck database, but robots.txt says
  `ai-train=no` and blocks ClaudeBot.
- **HSGuru / d0nkey.top.** Same `ai-train=no` and ClaudeBot block.
- **HSReplay.net.** Robots allow, but scripted requests get a Cloudflare
  challenge (403), and its deck data is a premium product.
- **Liquipedia.** robots.txt blocks ClaudeBot and other AI crawlers.
- **Kaggle "History of Hearthstone".** ~346k HearthPwn decks; same
  provenance as the Zenodo set, which supersedes it.

## Flesh and Blood

Player-made decklist sources, surveyed 2026-10-28 (write-up:
`fab_deck_sources.md`, next to this file). fabtcgmeta.com's Talishar
online decks (~800k, with card lists) are built as
`fabtcgmeta_online_decks/`.

- [ ] **Fabrary: needs the user's call.** The largest FaB deck site.
  Its robots.txt says `ai-train=no` (Cloudflare Content-Signal), and
  there's no published dataset of it, so using it means scraping it
  directly.
- [ ] **Snapshot fabtcgmeta periodically.** Its shards hold a rolling
  365-day window: new decks move from recent into archive, and decks
  over a year old drop out. A re-run skips shards already on disk, so a
  refresh must re-download both tiers (~2.3 h), ideally into a dated
  folder so older raw snapshots are kept. Run it at least every few
  months.
- [ ] **FaB Insights API (June 2025 onward).** Talishar sends every
  game, with both decks, here and to Fabrary. A free key is available
  from the maintainer (github.com/AnoAn/talishar-data-downloader).
  Adds Jun-Oct 2025, and keeps per-game history that fabtcgmeta drops.
- [ ] **cardvault gaps.** The binder is from the 2026-10-02 cardvault
  CSV. Some Talishar card names aren't in it (e.g. "Fire That Burns
  Within", "Knife Through Butter") and map to Unknown; refreshing
  `cardvault_fabtcg` and the FaB binder may pick them up.
