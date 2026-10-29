# Sample free RAM, python working set and python dedicated VRAM every 30 s
# until logs/<run>.finished exists. Usage: powershell -File sample_resources.ps1 <run>
param([string]$Run)
$R = "data/runs/2026-10-24_scale_profile"
$out = "$R/logs/$Run.resources.csv"
# Never overwrite an earlier sample file: a second sampler for the same run gets its own file
if (Test-Path $out) { $out = "$R/logs/$Run.resources.$(Get-Date -Format yyyyMMdd_HHmmss).csv" }
"time,free_ram_gb,python_ws_gb,python_vram_gb,total_vram_gb" | Out-File -Encoding utf8 $out
while (-not (Test-Path "$R/logs/$Run.finished")) {
  $os = Get-CimInstance Win32_OperatingSystem
  $py = Get-Process python -ErrorAction SilentlyContinue
  $ws = ($py | Measure-Object WorkingSet64 -Sum).Sum
  $pids = @($py | % { "pid_$($_.Id)_" })
  $vram = 0; $total = 0
  try {
    (Get-Counter "\GPU Process Memory(*)\Dedicated Usage" -ErrorAction Stop).CounterSamples | % {
      $s = $_; foreach ($p in $pids) { if ($s.InstanceName -like "$p*") { $vram += $s.CookedValue } } }
    $total = ((Get-Counter "\GPU Adapter Memory(*)\Dedicated Usage" -ErrorAction Stop).CounterSamples | Measure-Object CookedValue -Maximum).Maximum
  } catch {}
  "{0},{1:N2},{2:N2},{3:N2},{4:N2}" -f (Get-Date -Format s), ($os.FreePhysicalMemory/1MB), ($ws/1GB), ($vram/1GB), ($total/1GB) | Out-File -Append -Encoding utf8 $out
  Start-Sleep -Seconds 30
}
