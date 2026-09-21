# TODO: isotropic/games

## Cross-player attack effects are dropped from partial decks

**Where:** `game_log_parser.py` (`_parse_turn_block()` and its line matchers,
`_line_belongs_to_owner()`), `partial_deck.py` (`partial_deck_card_uuids()`).

**Problem:** Some continuation lines in a turn block name a *different* player
than the turn's owner, e.g. a Witch forcing an opponent to gain a Curse:

    Strategyst plays a Witch.
    ... luzifer851 gains a Curse.

`Turn` has a single `player_nick`, so attributing that gain to luzifer851's deck
would need two things the current data model lacks:

1. Each `CardQuantity` carrying its own subject nick (who gained/trashed it).
2. A global event ordering across turns (e.g. a `sequence_index` on `Turn`), so
   `partial_deck_card_uuids()` can collect "gains attributed to me" from *any*
   turn that precedes my checkpoint, not just my own turns.

**Current behavior (deliberate, documented interim decision):** any continuation
line naming an explicit nick other than the block's owner is skipped, same as
reveals/discards. Consequence: attack-forced gains (Curses, and any
other-player trash/gain, e.g. a Trader reaction) are missing from the victim's
partial deck. The partial deck is therefore approximately right, not exact.

**Affected metrics:** all five `Metric[GameLog]` metrics that build partial
decks (`mid_game_next_buy`, `mid_game_next_trashed_card`,
`mid_game_next_turn_action_count`, `mid_game_win_probability`,
`mid_game_deck_pair_winner`). Curse-heavy games are hit hardest.

**Decision (2026-09-19):** ship as-is. Roughly 80% accurate is good enough;
breadth of metrics is prioritized over fidelity of any single one. Revisit only
if partial-deck accuracy turns out to matter for a trained model.

**Fix sketch, if revisited:**
- Add `subject_nick` to `CardQuantity` (or wrap in an attributed-event type) and
  `sequence_index` to `Turn`.
- Replace `_line_belongs_to_owner()`'s reject branch with capturing the explicit
  nick as the subject.
- Change `partial_deck_card_uuids()` to walk all turns before the target
  turn's `sequence_index` and filter gained/trashed entries by subject nick.
- Re-check `NextTrashedCardMetric`: "trash on this turn" should probably stay
  owner-only.

## Measured accuracy of the partial deck (added at implementation time)

Each file's header lists every player's final deck ("[25 cards] 1 Bridge, ...").
Rebuilding it as 7 Coppers + 3 Estates + gained - trashed from the parsed log
matches exactly for **~55% of players (~65% ignoring Curses)** across 1,500
2013 games (157 players in 2010: 59% / 70%). Remaining misses, by rough size:

1. Cross-player gains (Curses, Mountebank's Copper + Curse, Governor's Silver
   for the opponent) - the drop described above; Curses alone are ~55% of
   the missing cards.
2. Cross-player trashes (Swindler/Thief/Pirate Ship victims) - same cause.
3. Ambassador: cards returned to the supply ("... returning it to the
   supply.") leave the owner's deck but are neither a trash nor a gain, so
   they show up as extra Coppers/Estates. Would need a new `cards_removed`
   field (kept out of `cards_trashed` so `NextTrashedCardMetric` isn't
   polluted by non-trash removals).
4. Trader: "... X reveals a Trader to gain a Silver instead of a Copper."
   (a replacement gain) is not parsed (~30 lines per 1,500 games).
5. Unattributed lines like "... The Silver is trashed." (Pirate Ship) name no
   player at all.

To re-measure after any parser change, rebuild each player's deck from
`GameLog.turns` and diff it against the header's `[N cards]` line.
