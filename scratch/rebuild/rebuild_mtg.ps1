# Rebuilds everything keyed on MTG deck ids after the multiset/every-decklist
# change (2026-10-06), one heavy job at a time. Old boxes are moved aside,
# not deleted. One log per step next to this script; summary.log has exit codes.
$ErrorActionPreference = "Continue"
Set-Location "G:\Projects\TradingCardGameEmbeddings"
Add-Type -Namespace Power -Name Native -MemberDefinition @'
[DllImport("kernel32.dll")]
public static extern uint SetThreadExecutionState(uint esFlags);
'@
[Power.Native]::SetThreadExecutionState([uint32]"0x80000001") | Out-Null

$python = "G:\Projects\venvs\tcg-rocm\Scripts\python.exe"
$env:PYTHONPATH = "."
$env:PYTHONUNBUFFERED = "1"
$dir = "scratch\rebuild"
$summary = Join-Path $dir "summary.log"
function Step($name, [string[]]$arguments) {
    "$(Get-Date -Format s) START $name" | Add-Content $summary
    & $python -u @arguments *>&1 | ForEach-Object { "$_" } | Out-File -FilePath (Join-Path $dir "$name.log") -Encoding utf8
    "$(Get-Date -Format s) END $name exit=$LASTEXITCODE" | Add-Content $summary
}

# 0. Move the stale boxes aside (old per-draft / presence-hash ids)
$aside = "data\_pre_multiset_backup"
New-Item -ItemType Directory -Force $aside | Out-Null
Move-Item "data\final\decks\mtg.db" "$aside\mtg.db" -ErrorAction Continue
Move-Item "data\metrics\seventeenlands\replay_data\deck_box.db" "$aside\replay_data_deck_box.db" -ErrorAction Continue
"$(Get-Date -Format s) moved old boxes to $aside" | Add-Content $summary

# 1-4. Canonical box, then every metric keyed on its deck ids
Step "deck_box_mtg" @("scripts/run_deck_box_ingestion.py", "--source", "seventeenlands_game_data")
Step "game_data" @("scripts/run_metrics.py", "--source", "seventeenlands_game_data")
Step "replay_data" @("scripts/run_metrics.py", "--source", "seventeenlands_replay_data")
Step "final_decks_mtg" @("scripts/run_metrics.py", "--source", "final_decks_mtg")

# 5. Stale splits (split files are reused whenever they exist)
Get-ChildItem "data\splits" -Filter "seventeenlands_game_data.*" | Remove-Item
Get-ChildItem "data\splits" -Filter "seventeenlands_replay_data.*" | Remove-Item
Get-ChildItem "data\splits" -Filter "final_decks.held_out_card_mtg*" | Remove-Item
Get-ChildItem "data\splits\contrastive" -Filter "contrastive.mtg*" | Remove-Item
"$(Get-Date -Format s) deleted stale MTG splits" | Add-Content $summary

# 6-7. Coverage over the whole corpus, then preflight
Step "coverage" @("scratch/rebuild/check_coverage.py")
Step "preflight" @("scripts/run_training.py", "scratch/rebuild/preflight_after_rebuild.yaml", "--check")

[Power.Native]::SetThreadExecutionState([uint32]"0x80000000") | Out-Null
"$(Get-Date -Format s) ALL DONE" | Add-Content $summary
