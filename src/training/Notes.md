# Notes (training): findings from the pre-training review

Findings behind `TODO.md`, recorded 2026-09-21: a dated snapshot, not
kept current (current status lives in `TODO.md`). Facts are measured on this
machine unless marked "from the web". Scratch scripts were not kept except
`scripts/verify_gpu.py`.

## 1. Hardware

- GPU: **AMD Radeon RX 6800**, 16 GiB, RDNA2 (`gfx1030`). Not NVIDIA, so no
  CUDA. Windows reports 4 GB for it only because the WMI field is 32-bit.
- CPU Ryzen 7 3700X (8 cores / 16 threads), 32 GB RAM, ~1.4 TB free on `G:`.
- Driver string `32.0.21045.5002`. No HIP SDK installed (not needed, see
  below). WSL is not set up (and ROCm under WSL reportedly does not work on
  the RX 6800, from the web).
- The project `venv/` has `torch 2.14.0+cpu`. It cannot use the GPU.

## 2. GPU verification (done)

- AMD's official Windows PyTorch wheels cover RX 7000/9000 only (from the
  web). RX 6000 works with AMD's *nightly* `gfx103X-dgpu` wheels, which are
  unofficial and come from a staging index last updated April 2026.
- Installed into a **separate venv**, `G:\Projects\venvs\tcg-rocm`
  (Python 3.12): `torch 2.10.0+rocm7.13.0a20260421`, plus numpy, pandas,
  pyarrow, transformers 5.17. The ROCm runtime ships inside the wheels.

  ```
  pip install "torch==2.10.0+rocm7.13.0a20260421" \
      --index-url https://rocm.nightlies.amd.com/v2-staging/gfx103X-dgpu/ \
      --extra-index-url https://pypi.org/simple
  ```
- `scripts/verify_gpu.py` passes 6/6: GPU seen as `gfx1030` with 16 GiB,
  matmul matches CPU (max error 2e-4), fp16 and bf16 autocast, SDPA,
  throughput, transformer training steps.

### Measurements (RX 6800)

Raw matmul, 4096x4096: fp32 15.1 TFLOPS, **fp16 25.2**, bf16 7.7. So use fp16
autocast with a GradScaler, not bf16.

ModernBERT-base (149M params), `attn_implementation="sdpa"`, real cards
(noise keys dropped), fp16 unless stated:

| Scenario | Result |
|---|---|
| Frozen, mixed-length batches (bs 16) | 28 cards/s |
| Frozen, length-sorted, 512-token cap | 117 cards/s |
| Frozen, length-sorted, 1024-token cap | 89 cards/s |
| Frozen, synthetic, tokens/s at 256 / 512 / 1024 | ~40k / ~28k / ~20k |
| Fine-tune, 256 tok, bs 16 / 32 | 51 cards/s (5.6 GiB) / 55 (9.2 GiB) |
| Fine-tune, 512 tok, bs 16 | 22 cards/s, 11.5 GiB |
| Fine-tune, 512 tok, bs 32 + grad checkpointing | 17 cards/s, 4.6 GiB |
| Fine-tune, 1024 tok, no checkpointing | OOM at bs 16 (bs 8 fits) |

What to take from this:

- **Padding dominated the first (mixed-batch) measurement.** Sorting by length
  gave ~4x. Mixed games/card sizes in one batch is the worst case.
- **No fast attention kernel.** torch warns "not compiled with memory
  efficient attention"; `sdpa` and `eager` are equal. Cost rises with length.
- **Windows spills VRAM into system RAM instead of raising OOM.** 512 tok,
  bs 32, no checkpointing peaked at 21 GiB on a 16 GiB card and dropped to
  3.2 cards/s with no error. The trainer's OOM handling will not see this;
  a sudden slowdown means memory overflow.
- fp16 with `GradScaler` gave finite losses throughout. The trainer has no
  GradScaler yet (its README lists it as "not built").
- `reference_compile` (an older ModernBERT kwarg) does not exist in
  transformers 5.x and raises `TypeError`. Loading with
  `attn_implementation="sdpa"` works as is.
- Sizing: a frozen LM at ~100 cards/s makes a 32-card step ~0.3 s in the LM,
  so ~10k steps is roughly an hour or two. Fine-tuning at 256 tokens is
  several times slower per card. Feasible for a first model, not fast.

## 3. Serialization problem (the biggest finding)

`serialize_card_to_json` does `json.dumps(card.raw_content)`, i.e. every raw
field, including image URLs, IDs, prices and legalities. Token counts of that
(300 sampled cards per game):

| Game | Cards | Median tokens (ModernBERT) | Over 512 |
|---|---|---|---|
| MTG | 38,741 | ~2,350 | 100% |
| FaB | 5,188 | ~740 | 100% |
| Pokemon | 16,803 | ~360 | 0% |
| STS2 | 578 | ~345 | 0% |
| Gwent | 1,261 | ~135 | 0% |

DistilBERT (the current default, 512 max, uncased) is worse still. With
`max_length=512`, the MTG rules text starts after token 512 in ~90% of cards,
so the encoder would mostly read URLs. A rough denylist (URL, ID, artist,
price, image keys) cuts MTG to a median of ~200 tokens (p90 ~470) and FaB to
~570. See section 7 for the follow-up analysis of general serialization
rules, using MTG as the stress test.

## 4. Text encoder choice

Recommendation: `answerdotai/ModernBERT-base` (Apache-2.0, 149M params, 8192
context, cased tokenizer, code-heavy pretraining). No open model is truly
tuned for JSON; this is the closest known. Runner-up:
`jinaai/jina-embeddings-v2-base-code` (needs `trust_remote_code`). Skip
`nomic-ai/modernbert-embed-base` (retrieval-tuned, needs prefixes). It loads
and runs on the GPU here.

## 5. Data coverage on disk

- **Card binders** (`data/final/cards`): MTG 38,741, Pokemon 16,803, FaB
  5,188, Gwent 1,261, STS2 578. None for Dominion or Hearthstone; Pitchstack
  has no ingestion stage.
- **Deck boxes** (`data/final/decks`): FaB 4,161, Gwent 60,197, STS2 7,800.
  Also `data/metrics/seventeenlands/game_data/deck_box.jsonl` (1,051 decks,
  KTK only) and `data/metrics/sts_gg/deck_box.jsonl` (1,004).
- **Metric parquets** (`data/metrics`): Gwent all 8 masks; STS 25 files;
  17lands MSH PremierDraft (2 draft files) and KTK TradSealed (11 game
  files). Nothing for the other ~34 17lands sets, 17lands replay data (139 GB
  raw), Play-Gwent, FaB decklists, Dominion or Isotropic.
- **Dojos** exist for Gwent, STS, all 26 17lands metrics, Dominion,
  Play-Gwent and contrastive. Only Gwent and STS were ever exercised (the old
  smoke test); no 17lands dojo has produced split files.
- Raw data: 461 GB, of which 17lands is 449 GB (draft 227, replay 139, game
  84). `data/raw` has no Dominion or Isotropic directory.
- `docs/metric_dojo_inventory.csv` is stale (paths and dojo names that do not
  exist). `scripts/smoke_test.py` calls the removed `training_data()` API.

## 6. Other code observations

- Dojo `name` is the parquet file stem and splits go to a flat
  `data/splits/<stem>_*.parquet`, so same-named metrics for different 17lands
  sets/formats collide (there is a TODO at `generic_dojo.py:73`). `Trainer`
  also raises on duplicate dojo names.
- Every generic dojo constructor re-runs `make_splits`, streaming the whole
  parquet (1.8 GB for `pool_conditioned_pick`), and with `rng_seed=None` a
  restart reshuffles the splits.
- The checkpointer writes a new directory on every new best round and never
  deletes old ones; each holds the full LM (via `encoder_only_state_dict`).
- The trainer's `max_batch_cost` counts cards (1 each), not tokens.
- No training driver script or config exists; `README.md` has only a snippet.
- Tests: 374 pass in `tests/training`, `tests/encoder_model`, `tests/dojos`
  (CPU venv).

## 7. Serialization rules of thumb (MTG as the stress test)

Method: took the 38,740-row Scryfall oracle dump
(`data/raw/scryfall/oracle-cards-20260912210156.jsonl`), applied candidate
rules one layer at a time, and measured ModernBERT token counts on a
sample of 9,685 cards (every 4th). Then checked the same generic rules on
FaB, Pokemon, STS2 and Gwent. A prototype of the rules lives only in the
session scratchpad (not in the repo); the rules below are what to build.

### Decision (2026-09-21): rules live in the ingestion stages

Simplification happens in each game's `CardIngestionStage`
(`src/data_refinement/card_binder/*/ingestion_stage.py`), not at
serialization time, so the binder's `raw_content` is already lean and the
serializer stays a plain compact `json.dumps(..., ensure_ascii=False,
separators=(",", ":"))` with no per-game profile. The durable, tracked
version of these rules is "What goes in `raw_content`" in
`src/data_refinement/card_binder/README.md`; this section keeps the
evidence. Changes from the list below: rule 4 (corpus-default omission) is
dropped entirely; rule 10 becomes "never remove a key a metric or dojo
reads" (README rule 4) since a stage now knows what is downstream; the
per-game profile idea is replaced by the stage itself.

### Where MTG's tokens go

Of an average 2,125 value-tokens per raw card, image URIs are 27%, related
URIs 14%, purchase URIs 9%, legalities 9%, `card_faces` 6% (mostly noise
inside each face), prices, IDs and other URIs ~20%. The fields that
describe the card as a game object (name, mana cost, type line, oracle text,
P/T, colors, keywords) are about 100 tokens, ~5%.

### The rules

1. **Render compactly.** `json.dumps(x, ensure_ascii=False,
   separators=(",", ":"))`. Default `json.dumps` costs +20% tokens
   (211 vs 175 median on the MTG result): it escapes non-ASCII (`—` for
   an em dash) and pads separators. Pretty-printing costs +50%. A plain
   `key: value` line format is the same size as compact JSON, so JSON costs
   nothing extra.
2. **Drop by value shape, not only by key name.** Remove any string that is
   a URL, a UUID, or an ISO date/timestamp, and any object left empty by
   that. Then drop keys named `id`/`*_id`/`*_ids`/`*_uri`/`*_url`. This
   single layer took MTG from a 2,350-token median to ~440 (and FaB 732 to
   ~275). Key-name rules alone miss things (Gwent's `artid`, which slips past
   an `_id` pattern), so keep per-game overrides.
3. **Drop empties.** `null`, `""`, `[]`, `{}` and `false` carry no
   information (absent means the same), but keep `0` (power 0 matters).
4. **Omit a value that equals the corpus-wide default; never drop the
   path.** Compute the modal value per path over the raw corpus (e.g.
   `lang="en"`, `object="card"`); omit a value only when it equals that
   default. Two traps found while testing:
   - Dropping the whole path when it is ~constant also destroyed real data
     (STS2 `keywords`/`tags`, Pokemon `retreatCost`, FaB `face_2_*`).
   - Measuring "constant" after falsy values were already dropped flags
     `digital`, `reserved` and `oversized` as constant `true` (their
     presence is the signal), so measure on the raw rows.
   Gain is small (~5%), so this is optional polish.
5. **Separate the game object from the printing.** A Scryfall oracle dump
   has one arbitrary printing per card, so `set`, `collector_number`,
   `artist`, `border_color`, `frame`, `finishes`, `games`, `prices`,
   `edhrec_rank`, `flavor_text` and similar describe that printing, not the
   card's rules. Drop them by default (MTG: 328 to 173 tokens median),
   per-game, and always keep any key a dojo masks or predicts (see rule 10).
6. **Collapse enum-valued dicts.** `legalities` is ~195 tokens as
   `{format: status}`; listing only the non-default statuses
   (`{"legal": [...], "banned": [...]}`) is ~40. Judgment call whether
   legalities belong in a card embedding at all (bans change over time).
7. **Bound every list, and drop cross-card references.** Token cards list
   every card that creates them in `all_parts` (Treasure: 12,681 tokens; 119
   cards have more than 8 entries, 21 more than 50). Beyond size, `all_parts`
   puts *other cards' names* in this card's text, which leaks held-out-tier
   cards into training inputs under the card holdout design. Default: drop
   it; if kept, cap at ~4-8 entries.
8. **De-duplicate equal strings within an object.** Keep the first, drop
   later fields whose (>= 4 char, non-numeric) string value is identical.
   No effect on MTG, but FaB drops 269 to 227 median (`face_1_true_name`,
   `face_1_typebox`, `face_1_rules_text` duplicate other fields) and STS2
   loses `type_key`/`rarity_key`/`description_raw`. Only do this for strings
   (dropping equal numbers or lists would erase real coincidences such as
   `colors` equal to `color_identity`).
9. **Order keys: short structured fields first, long free text and relation
   lists last.** Costs nothing, and if a text is truncated it only loses
   the tail. It conflicts with augmentation `Mod`s that shuffle key order,
   so apply the order to the base form and let a mod shuffle afterwards.
10. **Guard the dojo interface.** The serializer must be told the keys any
    dojo masks or predicts, never drop them, and never drop a value equal to
    the mask sentinel (`"[MASK]"`). Also mask correlated fields together
    (masking `set` while `set_name` remains leaks the answer).
11. **Normalize numbers:** `3.0` to `3`.
12. **Verify every game with a report,** run when a game is onboarded: token
    percentiles, the costliest keys, keys removed entirely, a check that no
    gameplay field was lost, and the count of identical serializations
    (must be 0; MTG had 0 in 9,685).

Presentation markup is a per-game extra: STS2 has `[gold]Attack[/gold]`,
FaB has `{br}` and `**bold**`.

### Result

Token counts (ModernBERT) with rules 1-8 and a curated MTG printing list, on
400 sampled cards per game:

| Game | Raw median | After median / p90 / p99 / max | Over 256 | Over 384 |
|---|---|---|---|---|
| MTG | 2,352 | 141 / 188 / 279 / 296 | 2% | 0% |
| FaB | 732 | 227 / 286 / 343 / 422 | 24% | 0% |
| Pokemon | 349 | 209 / 254 / 303 / 357 | 9% | 0% |
| STS2 | 342 | 148 / 201 / 262 / 806 | 1% | 0% |
| Gwent | 132 | 95 / 122 / 154 / 188 | 0% | 0% |

So a **384-token cap** covers essentially every card after these rules (256
would truncate 24% of FaB and 9% of Pokemon). Also, a 143-token median is
what the GPU measurements in section 2 want: frozen encoding at ~100+
cards/s and fine-tuning at ~50 cards/s become realistic.

## 8. Scryfall ingestion stage review

`src/data_refinement/card_binder/scryfall/ingestion_stage.py`, checked
against `data/raw/scryfall` and `data/final/cards/mtg.jsonl`.

**Loaded correctly.** The dump has 38,740 rows, all with a unique
`oracle_id`. The binder has 38,741 cards: 38,740 are byte-identical to their
dump rows, and there are no missing or duplicate cards. The extra card is
the `Unknown` sentinel (see below). Alias ledger: 38,740 Scryfall, 39,846
MTGO, 29,544 Gatherer, 11,862 Arena entries.

**Findings and suggestions:**

1. **Non-card objects are ingested as cards.** The stage deliberately does
   "no filtering by layout/type". The dump contains 2,243 `art_series`
   objects (fake "Card // Card" entries, no rules text) and 291 `front_card`
   objects, ~6.5% of the binder, plus 913 tokens, 87 emblems, 102 schemes,
   107 vanguards and 207 planar cards. These pollute masked-field dojos,
   the contrastive negative pool, and any `all_cards()` scan. Suggest
   skipping `art_series` and `front_card` at ingestion, and deciding
   deliberately about tokens/emblems/schemes/planar (keep but flag with a
   `layout`-based kind). Do not skip digital-only cards (2,132, Arena/MTGO
   only): 17lands is Arena data.
2. **`keep_longer_content` is the wrong merge policy for an authoritative
   feed.** It keeps the row with more JSON bytes, and those bytes are
   mostly URIs and prices. On a re-ingest of a newer dump, an errata'd
   (shorter) oracle text could lose to the stale row. `oracle_id` is a
   perfect natural key, so `keep_incoming` is likely right. Not triggered
   today (one dump ingested), so this is reasoning, not an observed bug.
3. **`raw_content` stores the whole row, including noise.** That is fine
   for a lossless store (the raw dump stays on disk anyway) provided the
   serializer filters (section 7). It also means content-size comparisons
   and any future field-based logic see the noise.
4. **The `Unknown` sentinel** (nocab_uuid `971d0fb4-...`, `raw_content = {}`,
   provenance `system`) is added to the MTG, FaB, Gwent and STS2 binders (not
   Pokemon). It serializes to `{}`. Anything iterating all cards (corpus-scan
   metrics such as `MaskedFieldMetric`, contrastive) should exclude it;
   worth confirming each consumer does.
5. **Docstring is stale:** it cites `oracle-cards-20260820090157.jsonl`;
   the file on disk is `oracle-cards-20260912210156.jsonl`.
6. Oracle-level vs printing-level: because the dump keeps one arbitrary
   printing per card, MTG `set`/`rarity` describe that printing. Do not build
   a set- or rarity-mask dojo on MTG expecting oracle-level meaning.

## 9. Encoder wiring (section B: max_length, model_kwargs, length bucketing)

`src/encoder_model/text_encoder.py`, `reference_singlecard_models.py`,
`reference_multicard_models.py`.

- `PretrainedTextEncoder`'s `max_length` default is now 384 (was 512),
  matching section 7's p99s across every game after the lean-`raw_content`
  work. It also takes `model_kwargs: dict | None`, forwarded to
  `AutoModel.from_pretrained` (e.g. `attn_implementation`). Loading
  ModernBERT-base with no `attn_implementation` specified already resolves
  to `sdpa` (checked directly: `AutoModel.from_pretrained(...).config._attn_implementation
  == "sdpa"`, no warnings), so `model_kwargs` is a general escape hatch, not
  something the reference presets need to set by default.
- Both reference-model files' `_DEFAULT_PRETRAINED_CHECKPOINT` switched from
  `distilbert-base-uncased` to `answerdotai/ModernBERT-base`.
- **Length bucketing**, inside `PretrainedTextEncoder.encode` only (no
  trainer/dojo change): a call over `length_bucket_size` texts (default 16)
  sorts by character length (a cheap proxy for token length - avoids
  tokenizing twice), splits into that many sorted sub-batches, tokenizes and
  forwards each separately (so each pads to its own, shorter max length),
  then reassembles into one result in original order. Reassembly pads every
  bucket's hidden_states/attention_mask up to the overall max sequence
  length via `index_copy_`, which is differentiable, so a trainable
  encoder's gradients still reach it. Sorting by CHARACTER length, not
  actually re-measuring token counts, is a deliberate simplification -
  works because tokens-per-character is fairly stable within one
  tokenizer/language.
  - Why 16 as a default: `HardwareLimits.max_batch_cost` is expected to
    start around 32 (see section A), and the bucket size needs to be
    smaller than a typical incoming batch or bucketing never triggers.
    This has NOT been tuned against a real dojo batch yet - revisit once
    real training numbers exist (section E of the TODO).
  - The original mixed-game benchmark in section 2 ("28 vs 117 cards/s")
    was an exaggerated worst case for THIS mechanism, since one dojo batch
    is normally one game, not several mixed together; the real win here is
    narrower (mainly separating a game's own long-tail outliers, e.g. MTG's
    p90 202 vs max 622 tokens) but still real.
  - `StaticEmbeddingTextEncoder` (the from-scratch baseline) was NOT given
    bucketing: it has no attention, so padding costs a few extra embedding
    lookups, not quadratic attention - nothing worth bucketing for.

### Masking-dojo leak check (also section B)

Checked every dojo that uses `MaskTargetKeyMod` against the real, lean
`raw_content` - today that's only `gwent_one` (8 dojos) and `dominiontabs`
(2 dojos); no other game has a masking dojo yet.

- **Found and fixed:** gwent.one `category` was exactly `"Leader"` on
  every `color == "leader"` card and never otherwise (confirmed against
  all 1260 cards, both directions) - pure duplication for those 42 cards,
  not real category information, so the ingestion stage now drops it for
  them (same treatment as an empty category). Closes a leak in
  `ColorMaskDojo`.
- **Found and fixed:** gwent.one `faction-duo`, present on 15 of 1260
  cards, always contains the true `faction` as its own prefix (e.g.
  faction `"syndicate"`, faction-duo `"syndicate_monster"`). Unlike
  `category`, this is NOT pure duplication - it names a real second
  faction - so the fix is at the dojo, not the stage: `FactionMaskDojo`
  now also masks `"faction-duo"` (unconditionally, via a second
  `MaskTargetKeyMod`; harmless on the 1245 cards that never had the key,
  which just gain a `"[MASK]"` they never had).
- **Checked, no leak:** Dominion `CostRegressionDojo` masks `cost`;
  `potcost`/`debtcost` remain unmasked but are different currencies, never
  literally equal to the coin-cost value being predicted (they correlate
  with it, which is legitimate signal, not a verbatim leak). Dominion
  `TypeMaskDojo` masks the whole `types` list value at once
  (`MaskTargetKeyMod` replaces the entire value with `"[MASK]"`), so list
  length/order changes from the lean-content work don't matter.
- This class of bug (a masked field's value restated in a sibling key)
  is exactly what card_binder/README.md's rule 4 warns about in the
  abstract; it needed checking against real data to actually find - the
  fix is recorded there with these two as concrete examples.

## Sources

- AMD ROCm Windows compatibility matrix:
  https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/windows/windows_compatibility.html
- gfx1030 on Windows setup notes:
  https://github.com/ssubedir/RCOm-windows-gfx1030/blob/main/README.md
- ROCm on WSL with RX 6800 (issue): https://github.com/ROCm/ROCm/issues/3371
- Open-source embedding model overview:
  https://www.bentoml.com/blog/a-guide-to-open-source-embedding-models
