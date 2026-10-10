#!/usr/bin/env bash
# 2026-10-09: after the full survey (step 32) ends, re-run replay_data with the
# file-backed deck box fix, then re-survey just the replay keys.
# Logs: logs/regen_2026-10-08/31b_*, 32b_*; START/END lines in summary.log.
cd /g/Projects/TradingCardGameEmbeddings || exit 1
export PYTHONPATH=. PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
PY=venv/Scripts/python.exe
LOG_DIR=logs/regen_2026-10-08
SUMMARY=$LOG_DIR/summary.log

# Wait for the full survey to finish
until grep -q "END   32 survey_dojo_baselines" "$SUMMARY"; do sleep 30; done

run_logged() {  # run_logged <num> <name> <args...>; samples all python memory every 30 s
  local num=$1 name=$2; shift 2
  local log="$LOG_DIR/${num}_${name}.log" mem="$LOG_DIR/${num}_${name}.mem.log"
  echo "$(date '+%F %T') START $num $name :: $*" >> "$SUMMARY"
  local t0; t0=$(date +%s)
  "$PY" -X faulthandler "$@" > "$log" 2>&1 &
  local pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    echo "$(date '+%T') $(tasklist //FI "IMAGENAME eq python.exe" //FO CSV //NH | tr '\n' ' ')" >> "$mem"
    sleep 30
  done
  wait "$pid"; local rc=$?
  echo "$(date '+%F %T') END   $num $name rc=$rc secs=$(( $(date +%s) - t0 )) tracebacks=$(grep -c Traceback "$log")" >> "$SUMMARY"
  return $rc
}

run_logged 31b metrics_seventeenlands_replay_data_rerun \
  scripts/run_metrics.py --source seventeenlands_replay_data || exit 1

run_logged 32b survey_replay_keys scripts/survey_dojo_baselines.py --keys \
  seventeenlands_replay_data.attacker_blocker_combat_outcome \
  seventeenlands_replay_data.average_turn_cast \
  seventeenlands_replay_data.cast_rate \
  seventeenlands_replay_data.combat_aggression_profile \
  seventeenlands_replay_data.combat_damage_push_through_rate \
  seventeenlands_replay_data.combat_kill_involvement_rate \
  seventeenlands_replay_data.discard_rate \
  seventeenlands_replay_data.turns_to_game_end_after_cast \
  seventeenlands_replay_data.tutor_target_rate

echo "$(date '+%F %T') ALL_DONE (replay rerun + replay re-survey)" >> "$SUMMARY"
