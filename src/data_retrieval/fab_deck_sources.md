# Flesh and Blood deck sources

Survey of 2026-10-28, looking for more FaB decklists than the 4,172
fabtcg.com tournament decks already in `data/final/decks/flesh_and_blood.db`.
robots.txt was checked for each site. The pick is at the end.

| Source | Decks | Card lists | AI / scraping signal | Verdict |
|---|---|---|---|---|
| fabtcgmeta.com online decks | ~800k (Oct 2025 onward) | yes | robots.txt allows `*`, GPTBot, ClaudeBot | **Built**: `fabtcgmeta_online_decks/` |
| fabtcgmeta.com paper decks | 3,875 | no (links to fabtcg.com) | as above | Duplicate of `fabtcg_decklists/` |
| fabtcg.com decklists | ~4k | yes | none | Already built (`fabtcg_decklists/`) |
| pitchstack.gg | 1,746 | yes, one call per version | none | Built (ids only); 99% fabtcg.com tournament mirrors |
| Fabrary (fabrary.net) | largest user deck site | yes (AppSync GraphQL) | **`ai-train=no`** Content-Signal | Not built: needs the user's call |
| FABREC (fabrec.com) | "500k decks analyzed" | unknown | **`Disallow: /` for `*`** | Not built |
| Talishar (talishar.net) | the games behind fabtcgmeta's online decks | via fabtcgmeta | `Disallow: /api/` | Use through fabtcgmeta |
| FaB Insights (fab-insights.azurewebsites.net) | every Talishar game, June 2025 onward | likely (Talishar sends both decks per game) | API key on request (free; Metafy supporters first) | Candidate: ask the maintainer |
| FaBDB (fabdb.net) | - | - | - | Dead: the domain no longer resolves |

## fabtcgmeta.com online decks (built)

fabtcgmeta is a meta-analysis site. Its "online" decklists are decks
played on Talishar (the fan-made online client), one per distinct list,
keyed by a 64-hex hash (`deckName`). The site's front end reads static
JSON from `https://data.fabtcgmeta.com/v1/`:

- `decklist/online/{archive,recent}/<first 3 hex chars>.json`: one
  shard per hash prefix (4,096 per tier). A shard maps
  `deckName -> {deckName, deckInfo, cards}`. `cards` has four groups
  (`heroWeaponEquipment`, `redPitchCards`, `yellowPitchCards`,
  `bluePitchCards`), each a list of `{name, count, pitch, ...}`.
  `deckInfo` holds hero, format (`CC-O`, `Silver Age-O`, `Blitz-O`,
  `Living Legend-O`) and date. Sampled shards: ~200 archive decks and
  ~15 recent ones each, ~600 KB uncompressed; archive and recent don't
  overlap.
- `decklists/online/<format slug>/_all.json` and `_heroes.json`: an
  index of the recent window only (912 CC decks), no card lists.

Card names are cardvault names, with differences the stage absorbs:
" // " where the binder has "||", letter case ("Consign To Cosmos"),
dropped apostrophes ("Swordmasters Path"). A name shared by several
pitch printings is narrowed by the entry's pitch, and a two-faced card
is also found by its front face. On 2,274 real decks (14 shards),
99.94% of copies matched; the misses were entries with empty names.
A few names seen elsewhere ("Fire That Burns Within") aren't in the
2026-10-02 cardvault CSV at all and map to Unknown.

fabtcgmeta's terms reserve copyright on its own analysis, articles and
site design; the decklists are players' Talishar decks.

## History: a rolling 365-day window

fabtcgmeta keeps only the last year. On 2026-10-28 the archive shards
(last modified 2026-10-22) held decks dated 2025-10-20 to 2026-09-24, and
its front end clamps every date window to 365 days. Older decks drop out
of the shards, so each download is a snapshot. Data before October 2025
is not available from fabtcgmeta.

At the end of each game, Talishar's server (`includes/functions.inc.php`,
open source) sends the result, with both decks, to just two places:

- **Fabrary**: has data from before June 2025; `ai-train=no`.
- **FaB Insights**: an API key is free on request
  (github.com/AnoAn/talishar-data-downloader). It serves one CSV per day
  and format, from June 2025 onward.

Talishar's own database is the only other copy.

## Fabrary (needs the user's call)

The largest FaB deck builder, with tournament results fed from Talishar.
Its robots.txt carries Cloudflare's Content-Signal
`search=yes,ai-train=no,use=reference`, an express reservation under
EU DSM Article 4. Not built. Same situation as HearthPwn behind the
Zenodo Hearthstone set, except no published research dataset exists for
FaB, so using it would mean scraping it directly.

## Sources searched and not found

No FaB decklist dataset on Kaggle, Zenodo or GitHub (the-fab-cube's
`flesh-and-blood-cards` is card data only). melee.gg returns 403 to
scripted requests.
