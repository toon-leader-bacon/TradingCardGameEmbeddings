# Runs the three 17lands metric families one after another (never in
# parallel: a game_data run beside a draft run ran out of memory before).
# Keeps the machine awake for the duration; the request is released when
# this script exits. One log per family next to this script.
$ErrorActionPreference = "Continue"
Set-Location "G:\Projects\TradingCardGameEmbeddings"

Add-Type -Namespace Power -Name Native -MemberDefinition @'
[DllImport("kernel32.dll")]
public static extern uint SetThreadExecutionState(uint esFlags);
'@
# ES_CONTINUOUS | ES_SYSTEM_REQUIRED: no idle sleep while this runs
[Power.Native]::SetThreadExecutionState([uint32]"0x80000001") | Out-Null

$python = "G:\Projects\venvs\tcg-rocm\Scripts\python.exe"
$env:PYTHONPATH = "."
$env:PYTHONUNBUFFERED = "1"
$logDir = "scratch\metric_runs"
$summary = Join-Path $logDir "summary.log"

foreach ($family in @("game_data", "draft_data", "replay_data")) {
    $log = Join-Path $logDir "$family.log"
    "$(Get-Date -Format s) START $family" | Add-Content $summary
    & $python -u scripts/run_metrics.py --source "seventeenlands_$family" *>&1 |
        ForEach-Object { "$_" } | Out-File -FilePath $log -Encoding utf8
    "$(Get-Date -Format s) END $family exit=$LASTEXITCODE" | Add-Content $summary
}

[Power.Native]::SetThreadExecutionState([uint32]"0x80000000") | Out-Null
"$(Get-Date -Format s) ALL DONE" | Add-Content $summary
