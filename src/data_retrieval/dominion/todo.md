# TODO — Dominion data sources

Research dump for onboarding Dominion (base set + expansions). Nothing
here is implemented yet — this is leads only, gathered by combining
the human's own digging with web research. Organized by the two flows
this project cares about (card ingestion, deck/metric data), plus a
catch-all for stats sources. Once any one of these turns into real
work, split it into its own subdirectory (mirroring
`sts2runs/`, `scryfall/`, etc.) and delete its section here.

## Card data (solved — just needs picking one + validating currency)

- [robinzigmond/Dominion-app](https://github.com/robinzigmond/Dominion-app)
  — `dominion_cards.csv`, 463 cards through Nocturne, richest field
  set of the options below (types, cost, setup/mat requirements,
  called cards). Best starting candidate.
- [quailman2101/dominion](https://github.com/quailman2101/dominion) — Not useful for this project (we want text of the card)
- [cypressf/dominion cards.json](https://github.com/cypressf/dominion/blob/master/cards.json) - Better but last updated 14 years ago :sad:
  — clean JSON, but stale: only 283 cards / 9 expansions, missing
  Adventures onward. Good schema reference, bad as the actual source.
- [wesbuck/DominionCardAPI](https://github.com/wesbuck/DominionCardAPI)
  ([Postman docs](https://documenter.getpostman.com/view/5603098/RWguxcDR))
  - Seems promising
  — live REST API wrapper over kingdom card data.
- Authoritative/current reference for validating whichever dump we
  pick: [dominionstrategy.com/all-cards](https://dominionstrategy.com/all-cards/)
  and its wiki mirror `dominionstrategy.miraheze.org` (standard
  MediaWiki, `api.php` export available; robots.txt only does routine
  bot-etiquette blocking, no AI/scrape prohibition).
- None of the GitHub dumps is verified current through the newest
  expansion — cross-check against the wiki before treating one as
  ground truth.

## Recommended kingdoms / card pools (not decks, but curated combos)

- [dominiondeck.com/games](https://www.dominiondeck.com/games/) — "2332
  named" curated kingdom setups (5 cards + theme name/title + tagged
  expansion sets, e.g. "Victory Dance", "Big Money"). Server-rendered,
  no public API, but has a `sitemap-index.xml` and no `robots.txt`
  disallow rules found (a `robots.txt` request 404'd through to the
  site's own 404 page, i.e. the route doesn't exist — treat as
  unrestricted but scrape politely, standard rate limit). Good
  phase-1 target: walk the sitemap, phase-2: pull each
  `/games/<slug>` page.
- Useful as a *pool* source (candidate kingdom compositions a metric
  or dojo could sample from) rather than a *played deck* source —
  nobody actually played these specific games, they're curated
  suggestions. Worth distinguishing from real decks in whatever schema
  ends up storing this.

## Real played decks / game logs (the hard one)

Confirmed dead ends first (per the human's own follow-up digging —
these were the three "obvious" places to look, all gone):

- **Isotropic archive.** `dominion.isotropic.org` itself is still up
  "down towards the bottom" of the page, but the actual
  `-summary.tar.bz2` / daily-tarball download links now 404 on the
  live site. The human only ever downloaded the *summary* logs
  historically, never the full game logs. The Wayback Machine URL
  found earlier (`web.archive.org/web/.../201303-summary.tar.bz2`)
  was only confirmed to resolve as a page reference, not verified to
  actually serve the binary — Wayback often archives a linking HTML
  page without archiving every large binary it links to. **Still
  worth a real check** (query the Wayback CDX API for
  `dominion.isotropic.org/gamelog/*` to see what was actually
  snapshotted, as opposed to just fetching one guessed URL), but treat
  as unconfirmed, not "found," until that's done with a plain HTTP
  client (my WebFetch tool refuses `web.archive.org` outright, so this
  needs a script, not agent tooling).
- **councilroom.com** went down in **mid-2022** (per the human,
  correcting my earlier vaguer "seems dead"). Example per-game URL
  pattern it used to serve, for reference if it ever turns up in an
  archive: `councilroom.com/game?game_id=game-20120519-133315-d9cb29f5.html`.
- **ceviri.me / Woodcutter's hosted logs** are also confirmed gone —
  e.g. `ceviri.me/woodcutter/89728623/plain/` (a specific log the human
  had bookmarked) no longer resolves, consistent with the domain being
  NXDOMAIN (found earlier this session).

None of the three public mirrors anyone used to point to survived.

### The live path: two-browser Selenium replay scraper (prototype, working)

Someone on the Discord posted a working prototype
(`~/Downloads/dominion_parser.py`, reviewed this session) that
reconstructs a full game log for *any* known game ID on the live
`dominion.games` site today — no dead domain involved, since it drives
the real site directly. Mechanism:

1. Two Selenium Chrome sessions, two logged-in dominion.games accounts
   (`login()` — note the script's password line is commented out, so
   as shipped it stops at typing the username and expects a human to
   finish login interactively, or relies on a persisted session).
2. `driver1` opens **New Table → Load Old Game → "Load from End"** for
   the target game ID — this replays the game to its final decision
   point rather than playing it out move-by-move.
3. A bot is added to fill the other seat, then `driver2` automatches
   into that same table (`Automatch` → finds `driver1`'s account by
   name in the friend-activity list) and readies up, which completes
   the replay and ends the game.
4. `get_log()` scrapes the `.game-log` DOM element's `innerHTML`,
   parses turn breaks (`new-turn-line` class) and nesting (`padding-left`
   style values) via regex, and reconstructs the full indented,
   turn-by-turn text log — the same shape as a manually-copy-pasted
   dominion.games log.
5. `load_game()` also has retry/recovery logic (`swap` param) for when
   automatch grabs the wrong seat, and a completion check comparing
   whether the Province pile is empty vs. inferring "maybe provinces"
   from remaining coins/pile count — a sanity flag on whether the
   replay actually reached a real game end.

This is the most concrete, currently-functional lead in this whole
list: it doesn't depend on any dead third-party domain, only on
`dominion.games` itself plus a game ID (like the `183560550` one
already in hand) and two accounts. Caveats before building on it:

- It's DOM/XPath scraping tied to the live Angular app's current
  markup (`ng-click="$ctrl.ok()"`, specific class names) — brittle to
  site updates, same fragility class as the dead Woodcutter grabber.
- Needs two real dominion.games accounts and real login automation
  (Selenium driving a live authenticated session) — heavier and more
  ToS-sensitive than anything else on this list; flag before running
  at any real volume rather than assuming it's fine to scale up.
- Only reconstructs one game per ID, sequentially, with multi-second
  waits per step (`time.sleep(1)`, `WebDriverWait` up to 10s) — fine
  for a targeted pull of known interesting game IDs, not obviously
  fine for bulk harvesting thousands of games without checking in
  first.

**DO NOT RUN THIS RIGHT NOW.** Per the human's read of the Discord
(where this script or one very similar to it was apparently already in
use): the dominion.games server owners explicitly asked the community
to pause this kind of automation, for two stated reasons — (1) a
server-side memory leak they're fixing and don't want extra load added
during the fix, and (2) they're considering shipping their own official
stats-tracking feature, which would make this whole approach obsolete
if/when it lands. Neither reason is "banned forever," but running this
now — even at small personal scale, even just to test — goes directly
against an explicit, recent ask from the people who run the service.
Revisit only once either (a) the server owners give an all-clear on the
memory-leak pause, or (b) they ship official tracking (in which case
this section is probably moot anyway). Until then this stays a
documented lead, not something to build or run.

### Other current/past-era leads (dombot / Dominion Scavenger)

1. **"Dominion Scavenger" (dominion.lauxnet.com) — real pipeline
   exists, currently unreachable.** Traced this from the
   `kierajreed/dombot` Discord bot: `dombot` is *not* a scraper, it's a
   read-only front-end over a private SQL table (`gameresults`,
   accessed via `config.shitdb.prod` in its source) populated by
   someone else's pipeline. That someone is `markus` (same person
   behind the stats write-up and Google Sheets below), and the
   pipeline is `dominion.lauxnet.com/scavenger/` (+ `/leaderboard/`,
   `/standings/`, `/live_leaderboard/`), confirmed via a
   [Shuffle iT forum thread](https://forum.shuffleit.nl/index.php?topic=2959.0).
   - Site was unreachable when checked (connection timeout on port 80,
     small personal DigitalOcean box) — retry later, it may just be
     down rather than gone.
   - Per the forum thread, this only covers **rated games**, and even
     the full pipeline's schema (reverse-engineered from `dombot`'s
     queries) stops at kingdom setup + final score breakdown
     (`pregame.gameParameters.setupInstructions.*`,
     `gameresult.playerResults[].score.parts[]`) — not an explicit
     full final-deck list. Real deck reconstruction still needs the
     turn-by-turn log underneath this.
   - **Best next step is social, not technical:** ask on the Discord
     whether `markus` (or whoever runs the bot/lauxnet site) would
     just share a data dump directly — far more productive than
     re-deriving or scraping their pipeline.
   - Current dominion.games games are replayable one-by-one via
     "New Table → Load Old Game → <game id>" (requires owning the
     relevant expansions in-client), and a community tool
     (`cevirici/dominion-woodcutter` + its
     [Tampermonkey grabber](https://greasyfork.org/en/scripts/38490-woodcutter-grabber))
     automated exactly this by driving the live Angular client
     (`tableService`, `replayService.getReplayInstructions()`, etc.) —
     i.e. real browser automation against an authenticated session,
     not a REST call. Both of that tool's own hosting domains
     (`ceviri.me` NXDOMAIN, `ceviri.net` parked/dead) are gone, so
     there's nothing live to reuse, but the mechanism is documented
     here if we ever want to rebuild it ourselves.

2. **2024 arXiv dataset — dead end unless the authors respond.**
   ["Dominion: A New Frontier for AI Research"](https://arxiv.org/abs/2405.06846)
   claims a 2,000,000+ real-game dataset pulled from dominion.games.
   No public download link, GitHub, Kaggle, or Zenodo entry found.
   Would need to email the authors directly.

3. **Self-play fallback — always available, zero scraping risk.**
   [pyminion](https://github.com/evansiroky/pyminion) and
   [rspeer/dominiate-python](https://github.com/rspeer/dominiate-python)
   (the latter powered "Provincial," historically the strongest known
   Dominion bot) are open-source Python rules engines supporting
   bot-vs-bot self-play. Running games locally generates real final
   decks (synthetic drafting outcomes, not human) with no ToS/scraping
   concerns at all — a solid source for `dojos/contrastive/` (winning
   deck vs. losing deck) even if we get nothing else on this list.

## Metrics / stats (several ready-to-use sources, no scraping fight needed)

- **Glicko card-strength ranking** — [Google Sheet](https://docs.google.com/spreadsheets/d/1CaVOd1pgAgmjJHXPM1tVMVnlJOLDZaq8BxjW4I1NI1E/edit?gid=0#gid=0),
  tabs per expansion (Cards, Events+Projects, Allies, Ways, Landmarks,
  Traits, Prophecies, Knights, Ruins, Rewards, Loot, Boons, Hexes).
  Columns: rank, rating, phi (uncertainty), comparisons, win%,
  beats-best/median/worst, expansion, Qvist cost category. Actively
  maintained (last updated 20-Jul-2026 at time of check) — this is the
  live, current equivalent of what councilroom.com used to provide
  before it died.
- **`markus`'s per-card / per-board stats** — starting sheet
  [here](https://docs.google.com/spreadsheets/d/1M2L7hcY3sbA33OwuZhgPYJWVlMFgJYBdK8cnkbJHmbo/edit?gid=0#gid=0),
  plus linked general/per-player sheets (e.g.
  [this one](https://docs.google.com/spreadsheets/d/1Qh4TTh-czArghea4s8ouqPzlxfvGGHx01y7B5TQp0ns/edit?gid=138555722#gid=138555722)).
  Tabs: buys/gains/trashes distributions, boards, opening gains,
  gain-1st, impact factor, outperformance vs. skill, first-player
  advantage, game-ending classification (Province/3-pile/resign).
  Methodology described in the human's forum-post writeup: ~24,000
  top-player games logged via Woodcutter, filtered to skilled players
  (mu ≥ 1.9). Rich, but it's aggregate stats, not raw decks — a
  `CorpusScanMetric`-shaped source, not a `Dojo`-ready deck list.
  A `loggedgames` tab reportedly lists the underlying game IDs used —
  worth pulling if we ever want to go back to raw logs for those
  specific games.
- Card strategy text on the wiki (one article per card, discussing
  synergies/counters) could seed a co-occurrence/synergy metric.

## Open questions before building anything

- Whether the two-browser Selenium replay scraper is worth productionizing
  is really a scope/ToS question for the human, not a technical one —
  it's the one lead that actually works today. Needs a decision before
  any real build: how many accounts, what volume, and how to source
  game IDs to feed it (the `loggedgames` tab mentioned under Metrics,
  or hand-picked ones like `183560550`).
- Isotropic tarball fetching needs a plain HTTP client through the
  Wayback CDX API to confirm anything was actually archived, before
  assuming the one guessed snapshot URL works — figure out the right
  tool/script for that outside of agent WebFetch, which refuses
  `web.archive.org` directly.
- Decide whether pursuing `dominion.lauxnet.com` further is worth it
  (retry when it's back up) vs. going straight to asking on Discord
  for a data dump.
- If self-play (pyminion/dominiate) becomes the primary deck source,
  that's arguably not `data_retrieval` at all (no external raw data
  being collected) — may belong in `data_refinement` or its own
  simulation-driven container instead. Worth raising when we get
  there rather than deciding now.
