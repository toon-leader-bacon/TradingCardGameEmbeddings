# Spring cleaning report (`cloud_cleanup`)

A review of the codebase against `PRINCIPLES.md`, `PATTERNS.md` and
`DOCUMENTATION_PRINCIPLES.md`. This is working notes for reviewing the
branch; delete it before merging.

- **Branched from:** `main` at `f8045b5`.
- **Size:** 18 commits.
- **Behavior:** unchanged everywhere, except one bug fix (commit
  "Fix metrics-private deck box paths").
- **On-disk formats:** unchanged (parquet schemas, binder JSONL, DeckBox
  SQLite schema, split file names).
- **Checks:** `./scripts/check.sh` passes at every commit. In this
  container, mypy needs `PATH=/usr/local/bin:$PATH` to see torch.
- **Tests:** 1,555 passed, 3 skipped. On `main`, one of them failed
  whenever `data/` was absent; it now skips.

## What changed

### Code

| Change | Principle |
|---|---|
| Split `DeckBox` (1,074 lines) into the public store and a private `DeckTables` table gateway (`_deck_tables.py`): one method per SQL statement, never commits. | 3: split files that outgrow themselves |
| 33 per-metric dojo wrappers repeated one of three identical ~20-line constructors. They now subclass `CardAverageMetricDojo` / `DeckLabelMetricDojo` / `MaskedFieldMetricDojo` (`dojos/generic/paired_metric_dojos.py`, Template Method) and only set `METRIC`. `METRIC` is typed against Protocols, so mypy rejects a class missing the needed ClassVars. -785 lines. | 2: duplication |
| `run_metrics.py`: three 17lands builder functions (~240 lines of copy-pasted constructor calls) became tuples of specs plus one builder. Constructed classes, arguments and output paths were verified identical before and after. | 2 |
| The 17lands name-matching policy (exact name, else split/MDFC front face) was copied byte-for-byte in 4 places. It is now `card_lookup.uuid_for_name_or_front_face()`. `metrics/TODO.md` said to do this once replay_data added the third copy; it had. | 2 |
| `play_gwent` downloader's private retry loop now uses `download_utils.call_with_retries` (same attempts and backoff). | 2 |
| Deleted dead code: `scripts/smoke_test.py` (called a removed API), `training/demo_training_loop.py`, `FileManagerCSV` and `SplitSampler` (no production consumers). The split-postfix helper was folded into its only consumer. | dead code |
| Moved `fabtcg_decklists/fragment_parsing.py` next to its only consumer in `deck_box/fabtcg_decklists/`. | 3: single consumer lives near its consumer |
| Renames: `DeckBoxDealer.py` → `deck_box_dealer.py`, `FileManagerParquet.py` → `file_manager_parquet.py` (Google style); `AliasLedger.resolve()` → `uuid_for()`, `card_resolution.py` → `card_names.py`, `_uuid_resolution.py` → `_row_values.py` (the "resolve" rule); `hash_utils.py` → `deck_ids.py` (named for what it holds). | 6, 7: naming, style guide |
| **Bug fix.** Metrics-private deck boxes were saved at `deck_box.jsonl` although `DeckBox` is SQLite now. With the old JSONL file on disk (Notes.md records two), `run_metrics.py --source sts_gg` scans everything and then crashes at `save()`, and the 17lands families crash on warm start. They are now `deck_box.db`; the old `.jsonl` files can be deleted. | correctness |
| New `scripts/report_metric_dojo_inventory.py` regenerates `docs/metric_dojo_inventory.csv` from the code (86 rows; it was entirely stale). A test fails if the CSV drifts. | docs 7: staleness is a bug |

### Docs

- **Rewritten to current state:** `src/README.md` (now a map with a
  status column), the root `README.md` (the two intros merged), and the
  READMEs for `data_retrieval/`, `data_refinement/`, `card_binder/`,
  `deck_box/`, `dojos/`, `training/` and `tests/`.
- **What that fixed:**
  - Classes and files that no longer exist (`DeckOutcomeMetric`,
    `DecklistCardsMetric`, `ItemEmbeddingStrategy`,
    `demo_training_loop.py`, `image_processing/`, `legacy/`).
  - Wrong counts.
  - Decision history.
  - Per-source detail sitting one or two levels too high. It became
    one-row-per-source tables that point at each stage's own docstring.
- **New READMEs** for main directories that had none:
  - `dojos/generic/` (9 subfolders).
  - `encoder_model/`: not "main" by the 4-subfolder/10-file rule, but it
    is the product.
- **TODO audit:**
  - Deleted as fully done: `data_refinement/TODO.md`,
    `metrics/play_gwent/TODO.md`.
  - Deleted as overcome by events: `deck_box/seventeenlands_game_data/TODO.md`
    (its "DeckBox is an in-memory dict" problem was fixed by the SQLite
    rewrite).
  - Pruned done items from: root, `data_retrieval/`, `metrics/`,
    `deck_box/`, `file_managers/` and `training/`.
  - `training/TODO.md` items re-checked against the code: MSH fix
    landed, split reuse done, stale files done.
  - Added `todo.md` to the empty `random_leader_mask_dojo/` placeholder.
- **Corrected:** `pitchstack.md` claimed the per-version card list
  endpoint was implemented. It isn't.
- **BRAINSTORM files:** stale "nothing built yet" headers corrected.
  Ideas left untouched.

## Deliberately not done: your call

Ordered by how much I'd recommend them.

1. **Parse, don't validate for raw rows (principle 1, highest
   priority).** 51 files use `Metric[dict]` and 145 functions take
   `row: dict`, so untyped raw rows travel well past the scanner
   boundary into every metric. A typed row per source (a dataclass
   built once in each scanner) would be the fix. It touches every
   metric family, so it's a design decision, not a cleanup.
2. **Delete `dojos/loss/win_loss_bce_loss.py`.** It's unused and broken
   (it passes Python lists and ±1.0 targets to `BCEWithLogitsLoss`), and
   `BceLoss` supersedes it. It was kept on purpose in an earlier
   session, so I left it.
3. **Per-metric wrappers don't accept a `DojoConfig`.** So two 17lands
   sets' same-named dojos (e.g. `deck_win_prediction` for KTK and MSH)
   can't be given distinct names or split prefixes, and `Trainer`
   raises on the duplicate. Adding `config: DojoConfig | None` to the
   three `paired_metric_dojos` bases would cover most wrappers in one
   place. (Tracked in `training/TODO.md` C.)
4. **The STS `"CARD."`-prefix + `get_by_alias` rule is repeated about 6
   times** (5 sts_gg metrics plus the sts_gg and sts2runs deck stages).
   It's the same shape of fix as `uuid_for_name_or_front_face()`, but
   the sts_gg README records that duplication as deliberate, pending a
   dedicated pass.
5. **Running-tally Template Method dedupe** across `CardAverageMetric`,
   `PackCardTallyMetric`, `GameCardAverageMetric` and several replay
   metrics (`metrics/TODO.md`). All three 17lands families now exist,
   so the shared shape can be read off real code.
6. **`Trainer`: pull mixed-precision stepping into its own
   collaborator.** Autocast, the scaler, unscale/clip/step and the
   overflow backoff would become a small `PrecisionPolicy` Strategy
   with fp32/fp16/bf16 variants, leaving `Trainer` pure orchestration.
   That code landed in the last commit on `main`, so I didn't churn it.
7. **Naming nits.** `FileManagerParquet` is named for an action
   ("Manager"); `ParquetSplitFiles` would say what it holds. Consider
   `Trainer.run()` → `train()`. Both are public, low value, and easy.
8. **Docstring history.** I removed decision-history narrative from
   every file I touched, but many module docstrings elsewhere still
   carry "CONFIRMED live on …", "this session" and "an earlier draft…"
   notes. `DOCUMENTATION_PRINCIPLES.md` rule 2 is written for READMEs.
   Say if you want it applied to docstrings too; it's a mechanical but
   large pass.
9. **`random_leader_mask_dojo/`**: decide whether it differs from
   `LeaderMaskedFromDeckDojo`, or delete it (see its `todo.md`).

## Needs data (couldn't verify here)

- Everything under "Data bugs" in `deck_box/TODO.md`.
- Regenerating the MSH draft metrics.
- Confirming a full 17lands deck box extraction completes under SQLite.
- The row-count and card-match-rate half of the inventory.
