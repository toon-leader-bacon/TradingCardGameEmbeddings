#!/usr/bin/env bash
# Regenerate binders -> deck boxes -> metrics (2026-10-08 rebuild).
# One step at a time (parallel heavy jobs OOM this machine).
# Per-step log: logs/regen_2026-10-08/<NN>_<name>.log
# Summary:      logs/regen_2026-10-08/summary.log  (START/END lines, rc, secs)
# Usage: scratch/regenerate_data.sh [first_step_number]   (resume from a step)

cd /g/Projects/TradingCardGameEmbeddings || exit 1
export PYTHONPATH=.
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
PY=venv/Scripts/python.exe
LOG_DIR=logs/regen_2026-10-08
mkdir -p "$LOG_DIR"
SUMMARY="$LOG_DIR/summary.log"
FIRST=${1:-1}
N=0

step() {  # step <name> <fatal:yes|no> <script> <args...>
  local name=$1 fatal=$2; shift 2
  N=$((N + 1))
  local num; num=$(printf "%02d" "$N")
  if [ "$N" -lt "$FIRST" ]; then return 0; fi
  local log="$LOG_DIR/${num}_${name}.log"
  echo "$(date '+%F %T') START $num $name :: $*" >> "$SUMMARY"
  local t0; t0=$(date +%s)
  "$PY" "$@" > "$log" 2>&1
  local rc=$?
  local secs=$(( $(date +%s) - t0 ))
  local tracebacks; tracebacks=$(grep -c "Traceback" "$log")
  echo "$(date '+%F %T') END   $num $name rc=$rc secs=$secs tracebacks=$tracebacks" >> "$SUMMARY"
  if [ "$rc" -ne 0 ] && [ "$fatal" = yes ]; then
    echo "$(date '+%F %T') ABORT: fatal step $name failed" >> "$SUMMARY"
    exit "$rc"
  fi
}

# 1. Card binders (fatal: everything downstream keys on them)
step card_binders yes scripts/run_card_binder_ingestion.py --all

# 2. Deck boxes (sts_gg skipped: its decks are in the carried StS2 box)
for src in fabtcg_decklists play_gwent isotropic pokemon_tcg spire_codex_runs seventeenlands_game_data; do
  step "deck_box_$src" no scripts/run_deck_box_ingestion.py --source "$src"
done

# 3. Small/medium metrics
for src in cardvault_fabtcg cross_game dominiontabs fabtcg_decklists gwent_one \
           hearthstonejson pokemon_tcg scryfall play_gwent play_gwent_guides sts_gg \
           final_decks_pokemon final_decks_flesh_and_blood final_decks_gwent \
           final_decks_dominion final_decks_slay_the_spire_2 final_decks_mtg; do
  step "metrics_$src" no scripts/run_metrics.py --source "$src"
done

# 4-5. Isotropic and StS2
for src in isotropic_summary isotropic_games spire_codex sts2_runs; do
  step "metrics_$src" no scripts/run_metrics.py --source "$src"
done

# 6. 17lands (longest)
for src in seventeenlands_game_data seventeenlands_draft_data seventeenlands_replay_data; do
  step "metrics_$src" no scripts/run_metrics.py --source "$src"
done

# 7. Baseline survey
step survey_dojo_baselines no scripts/survey_dojo_baselines.py

echo "$(date '+%F %T') ALL_DONE" >> "$SUMMARY"
