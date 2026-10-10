#!/usr/bin/env bash
# 17lands cleanup (2026-10-04), review before running.
# 1. Old <SET>/<Format>/<stem> layout dirs, superseded by the rerun (0.4 GB).
#    find_partitions() never reads them.
# 2. Empty (0-row) replay partitions left by the 2021-layout CSVs, which are now skipped.
# 3. Test-pollution splits from a dojo test (src/dojos/file_managers/TODO.md).
set -euo pipefail
cd /g/Projects/TradingCardGameEmbeddings
rm -rf "data/metrics/seventeenlands/draft_data/OM1"
rm -rf "data/metrics/seventeenlands/game_data/BLB"
rm -rf "data/metrics/seventeenlands/game_data/BRO"
rm -rf "data/metrics/seventeenlands/game_data/Cube_-_Powered"
rm -rf "data/metrics/seventeenlands/game_data/DFT"
rm -rf "data/metrics/seventeenlands/game_data/DMU"
rm -rf "data/metrics/seventeenlands/game_data/DSK"
rm -rf "data/metrics/seventeenlands/game_data/ECL"
rm -rf "data/metrics/seventeenlands/game_data/EOE"
rm -rf "data/metrics/seventeenlands/game_data/FDN"
rm -rf "data/metrics/seventeenlands/game_data/FIN"
rm -rf "data/metrics/seventeenlands/game_data/HBG"
rm -rf "data/metrics/seventeenlands/game_data/HOB"
rm -rf "data/metrics/seventeenlands/game_data/KTK"
rm -rf "data/metrics/seventeenlands/game_data/LCI"
rm -rf "data/metrics/seventeenlands/game_data/LTR"
rm -rf "data/metrics/seventeenlands/game_data/MH3"
rm -rf "data/metrics/seventeenlands/game_data/MKM"
rm -rf "data/metrics/seventeenlands/game_data/MOM"
rm -rf "data/metrics/seventeenlands/game_data/MSH"
rm -rf "data/metrics/seventeenlands/game_data/NEO"
rm -rf "data/metrics/seventeenlands/game_data/OM1"
rm -rf "data/metrics/seventeenlands/game_data/ONE"
rm -rf "data/metrics/seventeenlands/game_data/OTJ"
rm -rf "data/metrics/seventeenlands/game_data/PIO"
rm -rf "data/metrics/seventeenlands/game_data/SIR"
rm -rf "data/metrics/seventeenlands/game_data/SNC"
rm -rf "data/metrics/seventeenlands/game_data/SOS"
rm -rf "data/metrics/seventeenlands/game_data/TDM"
rm -rf "data/metrics/seventeenlands/game_data/TLA"
rm -rf "data/metrics/seventeenlands/game_data/TMT"
rm -rf "data/metrics/seventeenlands/game_data/WOE"
rm -rf "data/metrics/seventeenlands/replay_data/PIO"
rm -rf "data/metrics/seventeenlands/replay_data/attacker_blocker_combat_outcome/AFR"
rm -rf "data/metrics/seventeenlands/replay_data/attacker_blocker_combat_outcome/STX"
rm -rf "data/metrics/seventeenlands/replay_data/combat_aggression_profile/AFR"
rm -rf "data/metrics/seventeenlands/replay_data/combat_aggression_profile/STX"
rm -f data/splits/tutor_choice_rate_source_{train,validation,test}.parquet
echo done
