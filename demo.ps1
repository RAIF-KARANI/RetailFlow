<#
RetailFlow live demo (Windows PowerShell). docs/12_DEMO.md explains each step.

  .\demo.ps1                 # 90 s live demo: shop-front camera -> API -> dashboard
  .\demo.ps1 -Seconds 300 -Video data\caviar\OneStopEnter2front\OneStopEnter2front.mpg

What it does:
  1. starts PostgreSQL, the API, the dashboard and the RTSP server (Docker Compose);
  2. publishes a looped dataset video as a live RTSP camera (FFmpeg -> MediaMTX);
  3. runs the pipeline on that stream, sending events to the API;
  4. you watch http://localhost:3001 (the session is marked "live").
Stop early with Ctrl+C; the simulated camera is stopped either way.
#>
param(
    [int]$Seconds = 90,
    [string]$Video = "data\caviar\TwoEnterShop1front\TwoEnterShop1front.mpg",
    [string]$Scene = "scenes\caviar_front.json",
    [string]$Stream = "demo_front"
)
# Note: no $ErrorActionPreference = "Stop" here. In Windows PowerShell 5.1 that turns
# Docker's normal progress output on stderr into a fatal error; exit codes are checked instead.
$py = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $Video)) { throw "Video not found: $Video (see tools\download_data.md)" }

Write-Host "1/4 Starting database, API, dashboard and RTSP server..." -ForegroundColor Cyan
cmd /c "docker compose up -d --build db api dashboard >NUL 2>&1"
if ($LASTEXITCODE -ne 0) { throw "docker compose failed: is Docker Desktop running?" }
cmd /c "docker compose --profile streaming up -d mediamtx >NUL 2>&1"
if ($LASTEXITCODE -ne 0) { throw "could not start MediaMTX" }
for ($i = 0; $i -lt 30; $i++) {
    try { if ((Invoke-RestMethod http://127.0.0.1:8000/api/health).status -eq "ok") { break } } catch { Start-Sleep 1 }
}

Write-Host "2/4 Publishing $Video as rtsp://127.0.0.1:8554/$Stream ..." -ForegroundColor Cyan
$sim = Start-Process -PassThru -NoNewWindow $py -ArgumentList "tools\simulate_rtsp.py", "$Stream=$Video"
Start-Sleep 4

try {
    Write-Host "3/4 Open http://localhost:3001/cameras/$((Get-Content $Scene -Raw | ConvertFrom-Json).camera_id) and pick the 'live' session." -ForegroundColor Green
    Start-Process "http://localhost:3001"
    Write-Host "4/4 Running the pipeline for $Seconds s (Ctrl+C to stop)..." -ForegroundColor Cyan
    & $py run.py --source "rtsp://127.0.0.1:8554/$Stream" --scene $Scene --out "outputs\demo\$Stream" `
        --api http://127.0.0.1:8000 --duration $Seconds | Select-String '"in"|"out"|keeps_up|realtime_factor|lag_slope|processed_frac|"skipped"|avg_proc_ms|Outputs'
}
finally {
    # Stop the simulator's whole process tree (only it, not other FFmpeg instances). With a venv,
    # python.exe is a launcher that starts the real interpreter, so FFmpeg is a GRANDCHILD:
    # stopping only direct children left FFmpeg encoding in the background.
    cmd /c "taskkill /PID $($sim.Id) /T /F >NUL 2>&1"
    Write-Host "Simulated camera stopped. Services keep running: docker compose --profile streaming down" -ForegroundColor Yellow
}
