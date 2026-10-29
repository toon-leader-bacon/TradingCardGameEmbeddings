# Sample GPU engine utilization and python CPU use every 15 s until
# logs/<run>.finished exists. Usage: powershell -File sample_utilization.ps1 <run>
# gpu_compute_pct / gpu_3d_pct / gpu_copy_pct: summed over python processes' engines.
# py_cpu_pct: python processes' CPU time as a % of one core (1600 = all 16 threads).
param([string]$Run)
$R = "data/runs/2026-10-24_scale_profile"
$out = "$R/logs/$Run.utilization.csv"
# Never overwrite an earlier sample file: a second sampler for the same run gets its own file
if (Test-Path $out) { $out = "$R/logs/$Run.utilization.$(Get-Date -Format yyyyMMdd_HHmmss).csv" }
"time,gpu_compute_pct,gpu_3d_pct,gpu_copy_pct,py_cpu_pct" | Out-File -Encoding utf8 $out
while (-not (Test-Path "$R/logs/$Run.finished")) {
  $pids = @(Get-Process python -ErrorAction SilentlyContinue | % { "pid_$($_.Id)_" })
  $c = 0; $d = 0; $k = 0; $cpu = 0
  try {
    (Get-Counter "\GPU Engine(*)\Utilization Percentage" -ErrorAction Stop).CounterSamples | % {
      $s = $_
      if ($pids | ? { $s.InstanceName -like "$_*" }) {
        if ($s.InstanceName -like "*engtype_compute*") { $c += $s.CookedValue }
        elseif ($s.InstanceName -like "*engtype_3D*") { $d += $s.CookedValue }
        elseif ($s.InstanceName -like "*engtype_copy*") { $k += $s.CookedValue }
      } }
    $cpu = ((Get-Counter "\Process(python*)\% Processor Time" -ErrorAction Stop).CounterSamples | Measure-Object CookedValue -Sum).Sum
  } catch {}
  "{0},{1:F1},{2:F1},{3:F1},{4:F1}" -f (Get-Date -Format s), $c, $d, $k, $cpu | Out-File -Append -Encoding utf8 $out
  Start-Sleep -Seconds 15
}
